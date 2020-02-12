import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=str, default=None)
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--distribution', type=str, default='binary')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--n_hidden', type=int, default=2)
    parser.add_argument('--alpha_dim', type=int, default=1)
    parser.add_argument('--interval', type=int, default=1000)
    parser.add_argument('--n_sample', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--name', type=str, default='THROWAWAY')
    parser.add_argument('--log_path', type=str, default='log')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=10)
    parser.add_argument('--pause', action='store_true')
    return parser.parse_args()

