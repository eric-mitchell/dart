import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--n_hidden', type=int, default=3)
    parser.add_argument('--alpha_dim', type=int, default=1)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--n_sample', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--name', type=str)
    parser.add_argument('--log_path', type=str, default='log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pause', action='store_true')
    return parser.parse_args()

