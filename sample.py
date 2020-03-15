import argparse
import torch
from torchvision import utils
import matplotlib.pyplot as plt
from src.dart import DARTHMM
from src.tensor import TT


def run(args: argparse.Namespace):
    device = 'cuda'
    if args.model == 'dart':
        d = DARTHMM(784, 2048, 3, args.alpha_dim, 'binary', dropout=args.dropout).to(device)
    elif args.model == 'tt':
        d = TT(784, args.alpha_dim).to(device)
    d.eval()
    d.load_state_dict(torch.load(args.archive)['model'])

    samples = d.sample(256, device=device, a1=0)[0]
    samples = samples.view(256,1,28,28).cpu()
    img = utils.make_grid(samples, nrow=16).permute(1,2,0)
    plt.imshow(img)
    plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha_dim', type=int, default=4)
    parser.add_argument('--archive', type=str)
    parser.add_argument('--model', type=str, default='dart')
    parser.add_argument('--dropout', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(get_args())
