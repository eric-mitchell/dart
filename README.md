# DART: Deep Autoregressive Tensor Trains

## Getting started

The code requires Python version >=3.6. Package dependencies are listed in `requirements.txt`. Install with `pip install -r requirements.txt`.

## Training

Available arguments are listed in `src/args.py`. Example training commands are:

`python -m run --distribution binary --name binary_nolatents --device cuda:0 --log_path log`

`python -m run --distribution gaussian --name gaussian_nolatents --device cuda:0 --log_path log`

`python -m run --distribution gaussian --name gaussian_latents4 --device cuda:0 --log_path log --alpha_dim 4`