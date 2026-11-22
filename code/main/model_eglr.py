"""
Entropy-Gated Latent Recursion (EGLR).

Training-free test-time refinement for instruct LLMs. At high-entropy decision
tokens, iterate the frozen base model's top L transformer layers on the
current-token hidden state until the next-token distribution converges
(KL < tau_kl) or a cap (K_max) is reached.

No new parameters, no fine-tuning, no auxiliary modules. Drop-in for any
causal decoder LM with the `model.model.layers` structure (Llama / Qwen /
Gemma families).

Usage
-----
    from model_eglr import EGLRModel, load_eglr_model
    model = load_eglr_model(config)              # from YAML
    out = model.generate_eglr(input_ids, attention_mask=mask,
                              max_new_tokens=512, tokenizer=tok)
    # out["generated"], out["per_token_depths"], out["refinement_rate"]

This module is self-contained. It reuses only utils.get_logger and
config.CACHE_DIR from the rest of the code.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.utils import is_flash_attn_2_available

try:
    from config import CACHE_DIR
except Exception:
    CACHE_DIR = os.getenv("HF_HOME", None)

try:
    from utils import get_logger
except Exception:
    import logging
    def get_logger(name): return logging.getLogger(name)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EGLR model
# ─────────────────────────────────────────────────────────────────────────────

class EGLRModel(nn.Module):
    """
    Entropy-Gated Latent Recursion — a training-free wrapper around a frozen
    causal LM.

    Parameters
    ----------
    base_model_name : str
        HF checkpoint.
    L : int
        Number of top transformer layers to re-invoke during refinement.
        0 disables EGLR entirely (degenerates to plain greedy/sampled decoding).
    alpha : float
        Fusion weight in h = (1-α)·h_anchor + α·h_k. Small values (0.1-0.3)
        anchor the refinement close to the original hidden state.
    K_max : int
        Maximum recursion iterations per token. 3 is a sensible default.
    tau_entropy : float
        Entropy threshold (nats). Refinement is triggered only when the
        next-token distribution's entropy exceeds this value. Intended to
        fire on 3-8% of tokens for math CoT.
    tau_kl : float
        Early-exit threshold for KL(probs_k || probs_{k-1}). When the
        distribution stops moving, we commit.
    norm_match : bool
        If True, rescale h_k to match h_anchor's L2 norm before fusion.
        Simple manifold-calibration heuristic that mitigates the "post-norm
        output fed back into pre-norm input" mismatch at the top-L boundary.
    """

    def __init__(
        self,
        base_model_name,
        L: int = 2,
        alpha: float = 0.2,
        K_max: int = 3,
        tau_entropy: float = 1.0,
        tau_kl: float = 1e-3,
        norm_match: bool = True,
        kv_cache: str = "actual",
        torch_dtype: str = "bfloat16",
        device_map: str = "cuda",
    ):
        super().__init__()

        self.L           = int(L)
        self.alpha       = float(alpha)
        self.K_max       = int(K_max)
        self.tau_entropy = float(tau_entropy)
        self.tau_kl      = float(tau_kl)
        self.norm_match  = bool(norm_match)
        if kv_cache not in ("actual", "modified"):
            raise ValueError(f"kv_cache must be 'actual' or 'modified', got {kv_cache!r}")
        self.kv_cache    = kv_cache

        logger.info(f"Loading frozen base model for EGLR: {base_model_name}")
        logger.info(
            f"EGLR config — L={self.L} alpha={self.alpha} K_max={self.K_max} "
            f"tau_entropy={self.tau_entropy} tau_kl={self.tau_kl} "
            f"norm_match={self.norm_match}"
        )

        kwargs = {
            "torch_dtype": getattr(torch, torch_dtype),
            "device_map":  device_map,
        }
        if CACHE_DIR:
            kwargs["cache_dir"] = CACHE_DIR
        if "gpt2" not in base_model_name.lower():
            if is_flash_attn_2_available():
                kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled.")
            else:
                kwargs["attn_implementation"] = "sdpa"
                logger.info("Flash Attention 2 unavailable — using SDPA.")

        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)

        # EGLR is inference-only. Freeze everything.
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.base_model.eval()

        self.config       = self.base_model.config
        self.total_layers = self.config.num_hidden_layers
        if self.L > self.total_layers:
            raise ValueError(f"L={self.L} > total_layers={self.total_layers}")
        self.start_layer_idx = self.total_layers - self.L

        assert hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"), \
            "EGLR assumes base_model.model.layers (Llama/Qwen/Gemma decoder-only)"

        # Forward hook on the LAST layer to capture its pre-final-norm output.
        # This gives us the deepest pre-LN tensor (full depth + correct regime)
        # to use as the initial h_k — better than all_h[-2] which is one layer
        # short.
        self._post_last_layer_h = None
        def _capture_last_layer(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            self._post_last_layer_h = h
        self.base_model.model.layers[-1].register_forward_hook(_capture_last_layer)

        logger.info(
            f"Refining layers [{self.start_layer_idx}:{self.total_layers}) "
            f"of {self.total_layers} total"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internal: run top-L layers + final norm on a given hidden state
    # ─────────────────────────────────────────────────────────────────────

    def _apply_top_L(
        self,
        hidden_state,         # (B, 1, H) input to layer (N-L)
        past_kv,              # DynamicCache, pre-cropped at top-L
        position_ids,         # (B, 1)
        cache_position,       # (1,)
        position_embeddings,  # (cos, sin) or None
        attention_mask_4d,    # (B, 1, 1, total_len) or (B, total_len) or None
    ):
        """Apply decoder layers [start_layer_idx : total_layers], then final norm.

        Side effect: `past_kv` is updated with new K,V for the current token at
        each layer in [start_layer_idx, total_layers). Caller is responsible for
        cropping the cache before each invocation if repeated calls are needed.
        """
        h = hidden_state
        for i in range(self.start_layer_idx, self.total_layers):
            layer_outputs = self.base_model.model.layers[i](
                h,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_values=past_kv,      # v5 uses the plural name
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            # v4 decoder layers return (hidden_states, ...) tuples; v5 return the
            # tensor directly. Unwrap only when it actually is a tuple — indexing
            # a bare tensor with [0] would slice the batch dim and drop a rank,
            # producing the dim-3 broadcast mismatch we hit on 5.5.x.
            h = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs
        h_pre_norm  = h
        h_post_norm = self.base_model.model.norm(h)
        return h_pre_norm, h_post_norm

    def _crop_cache_top_L(self, cache, new_length):
        """Truncate only the top L layers of `cache` to `new_length` positions.

        Handles both DynamicCache APIs:
          - transformers >= 4.55: cache.layers[i] exposes .crop(max_length)
          - transformers <  4.55: cache.key_cache[i] / cache.value_cache[i] are
                                   tensor lists we can slice directly.
        """
        # ── New API (transformers >= 4.55) ──
        if hasattr(cache, "layers") and len(cache.layers) > 0 and hasattr(cache.layers[0], "crop"):
            for i in range(self.start_layer_idx, self.total_layers):
                if i < len(cache.layers):
                    cache.layers[i].crop(new_length)
            return

        # ── Legacy API (transformers < 4.55) ──
        if hasattr(cache, "key_cache"):
            for i in range(self.start_layer_idx, self.total_layers):
                if i < len(cache.key_cache):
                    kc, vc = cache.key_cache[i], cache.value_cache[i]
                    if kc is not None and kc.shape[-2] > new_length:
                        cache.key_cache[i]   = kc[:, :, :new_length, :].contiguous()
                        cache.value_cache[i] = vc[:, :, :new_length, :].contiguous()
            return

        raise RuntimeError(
            f"Unsupported DynamicCache API on {type(cache).__name__}: "
            "neither .layers[i].crop() nor .key_cache[i] available. "
            "Likely transformers version mismatch — try 4.45+."
        )

    def _snapshot_top_L_token(self, cache, pos):
        """Save K,V at sequence position `pos` for top-L layers.

        Used by Fix B to undo refinement's K/V corruption: snapshot the
        clean K/V written by the initial forward pass, run the refinement
        loop (which overwrites these slots iteratively), then restore.
        """
        snap = {}
        if hasattr(cache, "layers") and len(cache.layers) > 0 and hasattr(cache.layers[0], "crop"):
            for i in range(self.start_layer_idx, self.total_layers):
                if i < len(cache.layers):
                    layer = cache.layers[i]
                    K = getattr(layer, "keys", None)
                    V = getattr(layer, "values", None)
                    if K is None:
                        K = getattr(layer, "key_cache", None)
                        V = getattr(layer, "value_cache", None)
                    if K is not None and V is not None:
                        snap[i] = (K[:, :, pos:pos+1, :].clone(),
                                   V[:, :, pos:pos+1, :].clone())
            return snap
        if hasattr(cache, "key_cache"):
            for i in range(self.start_layer_idx, self.total_layers):
                if i < len(cache.key_cache):
                    K, V = cache.key_cache[i], cache.value_cache[i]
                    if K is not None:
                        snap[i] = (K[:, :, pos:pos+1, :].clone(),
                                   V[:, :, pos:pos+1, :].clone())
        return snap

    def _append_top_L_token(self, cache, snap):
        """Append snapped K,V to top-L layers (assumes layers were just cropped)."""
        if hasattr(cache, "layers") and len(cache.layers) > 0 and hasattr(cache.layers[0], "crop"):
            for i, (K_snap, V_snap) in snap.items():
                if i < len(cache.layers):
                    layer = cache.layers[i]
                    if hasattr(layer, "keys") and getattr(layer, "keys", None) is not None:
                        layer.keys = torch.cat([layer.keys, K_snap], dim=2)
                        layer.values = torch.cat([layer.values, V_snap], dim=2)
                    elif hasattr(layer, "key_cache") and getattr(layer, "key_cache", None) is not None:
                        layer.key_cache = torch.cat([layer.key_cache, K_snap], dim=2)
                        layer.value_cache = torch.cat([layer.value_cache, V_snap], dim=2)
            return
        if hasattr(cache, "key_cache"):
            for i, (K_snap, V_snap) in snap.items():
                cache.key_cache[i]   = torch.cat([cache.key_cache[i],   K_snap], dim=2).contiguous()
                cache.value_cache[i] = torch.cat([cache.value_cache[i], V_snap], dim=2).contiguous()

    def _get_position_embeddings(self, hidden_state, position_ids):
        """Fetch rotary (cos, sin). Tolerate architectures where rotary
        is inside each attention layer (older HF / certain families)."""
        rot = getattr(self.base_model.model, "rotary_emb", None)
        if rot is None:
            return None
        try:
            return rot(hidden_state, position_ids)
        except TypeError:
            # Some rotary_emb take only (x, seq_len) — fall back to None and
            # let each decoder layer compute rotary internally.
            return None

    def _build_refinement_attn_mask(self, full_attn_mask, total_len, dtype, device):
        """Build a 4D additive attention mask for single-token refinement queries.

        During refinement we feed one query token at position `total_len - 1`
        and attend against `total_len` KV positions. Some of those positions
        may be padding (when batched samples had different prompt lengths).
        We add a large negative to padded positions so they contribute ~0
        to the softmax.

        Shape returned: (B, 1, 1, total_len).

        For B=1 or fully-unpadded batches the returned mask is all-zero,
        functionally equivalent to None but safer to always pass.
        """
        # full_attn_mask is the 2D boolean/int mask (B, total_len)
        # carried through generate_eglr — 1 for real tokens, 0 for pads.
        min_val = torch.finfo(dtype).min
        # (B, total_len) -> (B, 1, 1, total_len)
        add_mask = (1.0 - full_attn_mask.to(dtype)) * min_val
        return add_mask[:, None, None, :]

    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sample(probs, temperature, top_p, top_k):
        """Sample one token per row from probs (B, V)."""
        if temperature != 1.0 and temperature > 0:
            logits = torch.log(probs.clamp_min(1e-12)) / temperature
            probs = F.softmax(logits, dim=-1)
        if top_k > 0:
            top_v, _ = torch.topk(probs, top_k, dim=-1)
            probs = torch.where(probs < top_v[..., -1:], torch.zeros_like(probs), probs)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if top_p < 1.0:
            sp, si = torch.sort(probs, dim=-1, descending=True)
            cs = torch.cumsum(sp, dim=-1)
            remove = cs > top_p
            remove[..., 0] = False
            remove[..., 1:] = remove[..., :-1].clone()
            sp = sp.masked_fill(remove, 0.0)
            sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probs = torch.zeros_like(probs).scatter_(-1, si, sp)
        return torch.multinomial(probs, 1).squeeze(-1)

    # ─────────────────────────────────────────────────────────────────────
    # Main generation loop
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_eglr(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=512,
        tokenizer=None,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """
        Autoregressive generation with entropy-gated latent recursion.

        Supports batched generation (B >= 1). For B > 1 prompts must be
        LEFT-PADDED by the caller (standard HF convention for generation),
        and `attention_mask` must correctly mark real vs pad positions
        (1 for real, 0 for pad). Right-padding will produce silently-
        wrong results because the KV cache would contain pad positions
        in the middle of each sequence.

        Per-sample refinement semantics are preserved: samples that never
        trigger refinement skip the loop entirely; samples that converge
        early stop contributing to depth counts even when other samples
        in the batch keep iterating.

        Returns
        -------
        dict
            "generated"           : LongTensor (B, T_total)
            "per_token_depths"    : list[list[int]] per-sample recursion depth at each step
            "per_token_entropies" : list[list[float]] per-sample entropy at each step
            "refinement_rate"     : float, fraction of generated tokens that triggered refinement
        """
        B = input_ids.size(0)
        device = input_ids.device

        if pad_token_id is None and tokenizer is not None:
            pad_token_id = tokenizer.pad_token_id
        if eos_token_id is None and tokenizer is not None:
            eos_token_id = tokenizer.eos_token_id
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generated = input_ids
        attn_mask = attention_mask.clone()
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        past_kv = DynamicCache()

        per_token_depths    = [[] for _ in range(B)]
        per_token_entropies = [[] for _ in range(B)]
        refinement_triggers = 0
        active_steps        = 0

        for step in range(max_new_tokens):
            is_first    = (step == 0)
            model_input = generated if is_first else generated[:, -1:]

            # ── Full forward pass ────────────────────────────────────────
            out = self.base_model(
                input_ids=model_input,
                attention_mask=attn_mask,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_kv = out.past_key_values

            # hidden_states has (N+1) entries in HF Llama/Qwen/Gemma:
            #   [0..N-1]: input to layer i (= output of layer i-1 for i > 0)
            #   [N]:      post-final-norm output
            # So input to layer (N-L) = hidden_states[start_layer_idx].
            all_h     = out.hidden_states
            h_anchor  = all_h[self.start_layer_idx][:, -1:, :]   # (B, 1, H), pre-layer-(N-L)
            logits_0  = out.logits[:, -1:, :]                    # (B, 1, V)
            probs_0   = F.softmax(logits_0.float(), dim=-1)
            logp_0    = F.log_softmax(logits_0.float(), dim=-1)
            entropy_0 = -(probs_0 * logp_0).sum(dim=-1).squeeze(-1)   # (B,)

            final_probs = probs_0
            depths      = [0] * B

            # Only refine active (non-finished) high-entropy samples
            needs_refine = (entropy_0 > self.tau_entropy) & (~finished)

            if self.L > 0 and needs_refine.any():
                # After the full forward pass, the cache has `total_len` entries
                # at ALL layers (including top-L). For refinement we repeatedly
                # re-run the top-L layers on the CURRENT token with different
                # inputs, so we strip the current token's K,V from top-L first
                # and re-add it on each iteration.
                total_len = past_kv.get_seq_length()
                pre_len   = total_len - 1

                # Fix B (only when kv_cache="actual"): snapshot the clean K/V
                # written by the initial forward pass. The refinement loop will
                # overwrite these slots; we'll restore them at the end so future
                # tokens attend against the original (not iteration-noise) K/V.
                # When kv_cache="modified", we keep the refined K/V instead.
                kv_snapshot = (
                    self._snapshot_top_L_token(past_kv, pos=pre_len)
                    if self.kv_cache == "actual" else None
                )

                position_ids   = torch.arange(pre_len, total_len, device=device).unsqueeze(0).expand(B, -1)
                cache_position = torch.arange(pre_len, total_len, device=device)
                pos_embeds     = self._get_position_embeddings(h_anchor, position_ids)

                # Flash Attention 2 strictly expects a 2D mask at the layer level.
                # SDPA/Eager expect the 4D additive float mask.
                attn_impl = getattr(self.base_model.config, "_attn_implementation", "eager")
                if attn_impl == "flash_attention_2":
                    layer_attn_mask = attn_mask
                else:
                    layer_attn_mask = self._build_refinement_attn_mask(
                        attn_mask, total_len, dtype=h_anchor.dtype, device=device,
                    )

                # Initialise h_k from the LAST layer's pre-final-norm output,
                # captured by the forward hook in __init__. This is the deepest
                # pre-LN tensor available — same regime as h_anchor AND full depth
                # (l35 included), unlike all_h[-2] which is one layer short.
                h_k          = self._post_last_layer_h[:, -1:, :]
                probs_prev   = probs_0
                probs_k      = probs_0.clone()
                # Per-sample "still iterating" flag — starts = needs_refine,
                # flips to False once that sample's KL falls below tau_kl.
                active        = needs_refine.clone()
                depths_tensor = torch.zeros(B, dtype=torch.long, device=device)

                for k in range(1, self.K_max + 1):
                    # Rewind top-L cache so current-token K,V is removed before re-run
                    self._crop_cache_top_L(past_kv, pre_len)

                    # Fuse — norm-match h_k to h_anchor's magnitude if enabled
                    if self.norm_match:
                        a_norm = h_anchor.norm(dim=-1, keepdim=True)
                        k_norm = h_k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        h_k_s  = h_k * (a_norm / k_norm)
                    else:
                        h_k_s = h_k
                    h_input = (1.0 - self.alpha) * h_anchor + self.alpha * h_k_s

                    # Re-run the top-L layers (frozen, no new parameters).
                    # We run on the full batch even though some samples have
                    # already converged or never triggered — their results will
                    # be masked off below. Wasted compute is bounded and small.
                    h_new_pre, h_new_post = self._apply_top_L(
                        h_input, past_kv, position_ids, cache_position,
                        pos_embeds, layer_attn_mask,
                    )

                    # lm_head expects post-norm; next-iter h_k stays pre-norm
                    logits_k = self.base_model.lm_head(h_new_post)
                    probs_k_new = F.softmax(logits_k.float(), dim=-1)
                    logp_k_new  = F.log_softmax(logits_k.float(), dim=-1)

                    # KL(probs_k || probs_prev) per-sample
                    log_prev = probs_prev.clamp_min(1e-12).log()
                    kl = (probs_k_new * (logp_k_new - log_prev)).sum(dim=-1).squeeze(-1)  # (B,)

                    # For samples still active, commit this iteration's state.
                    # For already-converged / never-triggered samples, keep the
                    # previous probs/h_k unchanged (no-op semantically).
                    active_3d = active.view(-1, 1, 1)
                    h_k        = torch.where(active_3d, h_new_pre,  h_k)
                    probs_k    = torch.where(active_3d, probs_k_new, probs_k)
                    probs_prev = probs_k

                    # Bump depth counter for samples that were active this round
                    depths_tensor = depths_tensor + active.long()

                    # Mark samples converged this round as inactive
                    converged = active & (kl < self.tau_kl)
                    active = active & ~converged

                    # Everyone done — stop early
                    if not active.any():
                        break

                # Commit refined probs only for samples that needed refinement.
                mask_expanded = needs_refine.view(-1, 1, 1).expand_as(probs_0)
                final_probs   = torch.where(mask_expanded, probs_k, probs_0)
                refinement_triggers += int(needs_refine.sum().item())
                depths = depths_tensor.tolist()

                # Fix B: restore the original K/V (only when kv_cache="actual").
                # When kv_cache="modified", we leave the refined K/V in the cache
                # so future tokens attend against the deliberated state.
                if self.kv_cache == "actual":
                    self._crop_cache_top_L(past_kv, pre_len)
                    self._append_top_L_token(past_kv, kv_snapshot)

            # ── Sample next token ────────────────────────────────────────
            # Fast path: top_k=1 forces argmax regardless of temperature/top_p,
            # so skip the sampling machinery entirely. Same output, less compute.
            if do_sample and temperature > 0 and top_k != 1:
                next_tok = self._sample(final_probs.squeeze(1), temperature, top_p, top_k)
            else:
                next_tok = final_probs.squeeze(1).argmax(dim=-1)

            if pad_token_id is not None:
                next_tok = torch.where(
                    finished, torch.full_like(next_tok, pad_token_id), next_tok
                )

            generated = torch.cat([generated, next_tok.unsqueeze(-1)], dim=-1)
            attn_mask = torch.cat(
                [attn_mask, (~finished).long().unsqueeze(-1)], dim=-1
            )

            for b in range(B):
                if not finished[b]:
                    per_token_depths[b].append(depths[b])
                    per_token_entropies[b].append(float(entropy_0[b].item()))
                    active_steps += 1

            # EOS check
            if eos_token_id is not None:
                eid_list = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
                for eid in eid_list:
                    finished = finished | (next_tok == eid)

            if finished.all():
                break

        return {
            "generated":           generated,
            "per_token_depths":    per_token_depths,
            "per_token_entropies": per_token_entropies,
            "refinement_rate":     refinement_triggers / max(active_steps, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_eglr_model(config):
    """Construct an EGLRModel from a parsed Config object."""
    eg = config.eglr
    return EGLRModel(
        base_model_name=config.model.name,
        L           = getattr(eg, "L",           2),
        alpha       = getattr(eg, "alpha",       0.2),
        K_max       = getattr(eg, "K_max",       3),
        tau_entropy = getattr(eg, "tau_entropy", 1.0),
        tau_kl      = getattr(eg, "tau_kl",      1e-3),
        norm_match  = getattr(eg, "norm_match",  True),
        kv_cache    = getattr(eg, "kv_cache",    "actual"),
        torch_dtype = config.model.torch_dtype,
        device_map  = config.model.device_map,
    )