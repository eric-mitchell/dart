import torch
import argparse
import matplotlib.pyplot as plt

from src.tensor import TT


def run(args: argparse.Namespace):
    if args.model == 'tt':
        model = TT(784, args.alpha_dim)
    elif args.model == 'dart':
        model = DART()
    else:
        raise NotImplementedError()

    model.load_state_dict(torch.load(args.path)['model'])

    if args.model == 'tt':
        p_alpha = model._log_p_alpha().exp()
        entropy = -(p_alpha * p_alpha.log()).sum(-1)
        image = entropy.detach().squeeze().numpy().reshape((28,28))
        plt.imshow(image)
        plt.colorbar()
        plt.show()
        
    import pdb; pdb.set_trace()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--alpha_dim', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    run(get_args())
