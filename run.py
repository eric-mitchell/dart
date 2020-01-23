import numpy as np
import torch
import argparse

from src.args import get_args


def run(args: argparse.Namespace):
    seed = args.seed if args.seed is not None else instance_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    run(get_args)
