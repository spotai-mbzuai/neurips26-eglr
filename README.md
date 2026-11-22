# Entropy-Gated Latent Recursion

A training-free decoding method that re-applies a frozen language model's top layers at uncertain tokens, giving a second, deterministic way to diversify reasoning rollouts beside temperature sampling.

## In simple words

To reason better at test time, language models usually sample many answers with some randomness (temperature) and pick the best. We add a second knob that needs no randomness: at tokens where the model is unsure, we run its top L layers again, up to a fixed number of times, until the next-token prediction settles. Different choices of L solve different problems, so combining L with temperature gives a larger and more useful pool of answers at almost the same cost per answer. On MATH-500 with Qwen2.5-3B-Instruct, picking the best answer from this combined pool reaches 91.6%, 8.2 points above temperature alone; we test 8 models and 6 math benchmarks.

## Setup

```bash
git clone https://github.com/spotai-mbzuai/neurips26-eglr.git
cd neurips26-eglr/code
pip install -r requirements.txt   # Python 3.11
```

Run every script from `code/`; see [code/README.md](code/README.md).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{bhattacharjee2026entropy,
  title={Entropy-Gated Latent Recursion},
  author={Bhattacharjee, Soham and Chauhan, Dushyant Singh and Lahlou, Salem and Takac, Martin and Lukas, Nils},
  journal={arXiv preprint arXiv:2606.16620},
  year={2026}
}
```

## License

Free for research and non-commercial use, provided you cite the paper above. Commercial use requires permission from the authors. See [code/LICENSE](code/LICENSE).
