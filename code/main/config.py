import os
import sys
import yaml
import argparse
from dotenv import load_dotenv

def _setup_hf_cache():
    """Set HuggingFace cache + load .env from config BEFORE importing HF libraries."""
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        env_path = config.get('parameters', {}).get('env_path')
        if env_path:
            load_dotenv(env_path)

        cache_dir = config.get('parameters', {}).get('cache_dir')
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
            hf_home = os.path.dirname(cache_dir)
            xdg_cache = os.path.dirname(hf_home)

            os.environ["HF_HOME"] = hf_home
            os.environ["HF_DATASETS_CACHE"] = cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
            os.environ["XDG_CACHE_HOME"] = xdg_cache
            os.environ["TRANSFORMERS_CACHE"] = cache_dir

            return cache_dir
    return None

# Call this IMMEDIATELY on import
CACHE_DIR = _setup_hf_cache()

class Config:
    """Simple class to convert dictionary to object with dot notation."""
    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    params = config.get('parameters') or {}
    if params.get('cache_dir'):
        params['cache_dir'] = os.path.expanduser(params['cache_dir'])
    return config

def parse_args():
    """Parses CLI arguments and returns args alongside the parsed configuration."""
    parser = argparse.ArgumentParser(description="RecursiveDiST Training for GSM8K")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode")
    parser.add_argument("--eval-base", action="store_true", help="Run evaluation on the base model without checkpoints")
    parser.add_argument("--train-base", action="store_true", help="Run plain LoRA CoT finetuning without belief modules")
    parser.add_argument("--output-format", type=str, default="float", choices=["int", "float"], help="Output format for numeric comparison")
    parser.add_argument("--out-store", dest="out_store", action="store_true", default=True, help="Store generations to disk")
    parser.add_argument("--no-out-store", dest="out_store", action="store_false", help="Do not store generations to disk")
    parser.add_argument("--verbose-cache", action="store_true", help="Enable cache verification logging")

    # CLI overrides for config
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--model_is_instruct", type=lambda x: x.lower() == 'true', default=None,
                        help="Whether the model is an instruct model (affects prompt formatting)")
    parser.add_argument("--lora_enabled", type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_bias", type=str, default=None)
    parser.add_argument("--lora_task_type", type=str, default=None)
    parser.add_argument("--lora_target_modules", type=str, default=None, help="Comma-separated list")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--val_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--wandb_path", type=str, default=None)
    parser.add_argument("--logs_path", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--train_max_length", type=int, default=None)
    parser.add_argument("--eval_max_length", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--val_split_size", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--do_sample", type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--eval_checkpoint", type=str, default=None)
    parser.add_argument("--belief_dim", type=int, default=None)
    parser.add_argument("--belief_layers", type=int, default=None)
    parser.add_argument("--max_recursion_depth", type=int, default=None)
    parser.add_argument("--kl_threshold", type=float, default=None)
    parser.add_argument("--log_alpha_every_n_steps", type=int, default=None)
    parser.add_argument("--alpha_learning_rate", type=float, default=None)
    parser.add_argument("--belief_n_heads", type=int, default=None)
    parser.add_argument("--belief_ff_dim", type=int, default=None)
    parser.add_argument("--belief_dropout", type=float, default=None)
    parser.add_argument("--system_train", type=str, default=None)
    parser.add_argument("--system_eval", type=str, default=None)

    args = parser.parse_args()

    # Load configuration
    config_dict = load_config(args.config)

    # Apply CLI overrides
    if args.cache_dir is not None: config_dict['parameters']['cache_dir'] = args.cache_dir
    if args.model is not None: config_dict['model']['name'] = args.model
    if args.torch_dtype is not None: config_dict['model']['torch_dtype'] = args.torch_dtype
    if args.device_map is not None: config_dict['model']['device_map'] = args.device_map
    if args.model_is_instruct is not None: config_dict['model']['is_instruct'] = args.model_is_instruct
    if args.lora_enabled is not None: config_dict['lora']['enabled'] = args.lora_enabled
    if args.lora_r is not None: config_dict['lora']['r'] = args.lora_r
    if args.lora_alpha is not None: config_dict['lora']['lora_alpha'] = args.lora_alpha
    if args.lora_dropout is not None: config_dict['lora']['lora_dropout'] = args.lora_dropout
    if args.lora_bias is not None: config_dict['lora']['bias'] = args.lora_bias
    if args.lora_task_type is not None: config_dict['lora']['task_type'] = args.lora_task_type
    if args.lora_target_modules is not None: config_dict['lora']['target_modules'] = args.lora_target_modules.split(',')
    if args.num_epochs is not None: config_dict['training']['num_epochs'] = args.num_epochs
    if args.batch_size is not None: config_dict['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None: config_dict['training']['learning_rate'] = args.learning_rate
    if args.weight_decay is not None: config_dict['training']['weight_decay'] = args.weight_decay
    if args.max_grad_norm is not None: config_dict['training']['max_grad_norm'] = args.max_grad_norm
    if args.warmup_ratio is not None: config_dict['training']['warmup_ratio'] = args.warmup_ratio
    if args.max_steps is not None: config_dict['training']['max_steps'] = args.max_steps
    if args.val_steps is not None: config_dict['training']['val_steps'] = args.val_steps
    if args.logging_steps is not None: config_dict['training']['logging_steps'] = args.logging_steps
    if args.output_dir is not None: config_dict['training']['output_dir'] = args.output_dir
    if args.checkpoint_path is not None: config_dict['training']['checkpoint_path'] = args.checkpoint_path
    if args.wandb_path is not None: config_dict['training']['wandb_path'] = args.wandb_path
    if args.logs_path is not None: config_dict['training']['logs_path'] = args.logs_path
    if args.patience is not None: config_dict['training']['patience'] = args.patience
    if args.grad_accum is not None: config_dict['training']['grad_accum'] = args.grad_accum
    if args.dataset_name is not None: config_dict['dataset']['name'] = args.dataset_name
    if args.train_max_length is not None: config_dict['dataset']['train_max_length'] = args.train_max_length
    if args.eval_max_length is not None: config_dict['dataset']['eval_max_length'] = args.eval_max_length
    if args.samples is not None: config_dict['dataset']['samples'] = args.samples
    if args.val_split_size is not None: config_dict['dataset']['val_split_size'] = args.val_split_size
    if args.max_new_tokens is not None: config_dict['generation']['max_new_tokens'] = args.max_new_tokens
    if args.temperature is not None: config_dict['generation']['temperature'] = args.temperature
    if args.top_p is not None: config_dict['generation']['top_p'] = args.top_p
    if args.top_k is not None: config_dict['generation']['top_k'] = args.top_k
    if args.do_sample is not None: config_dict['generation']['do_sample'] = args.do_sample
    if args.test_batch_size is not None: config_dict['generation']['test_batch_size'] = args.test_batch_size
    if args.eval_checkpoint is not None: config_dict['generation']['eval_checkpoint'] = args.eval_checkpoint
    
    if 'recursive_dist' in config_dict:
        if args.belief_dim is not None: config_dict['recursive_dist']['belief_dim'] = args.belief_dim
        if args.belief_layers is not None: config_dict['recursive_dist']['belief_layers'] = args.belief_layers
        if args.max_recursion_depth is not None: config_dict['recursive_dist']['max_recursion_depth'] = args.max_recursion_depth
        if args.kl_threshold is not None: config_dict['recursive_dist']['kl_threshold'] = args.kl_threshold
        if args.log_alpha_every_n_steps is not None: config_dict['recursive_dist']['log_alpha_every_n_steps'] = args.log_alpha_every_n_steps
        if args.alpha_learning_rate is not None: config_dict['recursive_dist']['alpha_learning_rate'] = args.alpha_learning_rate
        if args.belief_n_heads is not None: config_dict['recursive_dist']['belief_n_heads'] = args.belief_n_heads
        if args.belief_ff_dim is not None: config_dict['recursive_dist']['belief_ff_dim'] = args.belief_ff_dim
        if args.belief_dropout is not None: config_dict['recursive_dist']['belief_dropout'] = args.belief_dropout
        
    if args.system_train is not None: config_dict['prompts']['system_train'] = args.system_train
    if args.system_eval is not None: config_dict['prompts']['system_eval'] = args.system_eval

    config = Config(config_dict)
    return args, config

#---------------------------
# DEPENDENCIES
#---------------------------
# torch
# transformers
# datasets
# tqdm
# pyyaml
# peft
# wandb
# python-dotenv
# accelerate

# # 1. Create and activate the conda environment

# conda create -n eglr python=3.12 -y

# conda activate eglr

# # 2. Install all required dependencies

# pip install torch transformers datasets tqdm pyyaml peft wandb python-dotenv accelerate