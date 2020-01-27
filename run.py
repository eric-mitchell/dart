import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import argparse
import os
import random

from src.args import get_args
from src.dart import DART


def get_mnist_data(device, batch_size: int = 128, binary: bool = False):
    preprocess = transforms.ToTensor()
    if binary:
        preprocess = transforms.Compose([
            preprocess,
            lambda x: x > 0.5,
            lambda x: x.float()
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=preprocess),
        batch_size=batch_size,  # Using a weird batch size to prevent students from hard-coding
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=preprocess),
        batch_size=batch_size,
        shuffle=True)

    # Create pre-processed training and test sets
    X_train = train_loader.dataset.train_data.to(device).reshape(-1, 784).float() / 255
    y_train = train_loader.dataset.train_labels.to(device)
    X_test = test_loader.dataset.test_data.to(device).reshape(-1, 784).float() / 255
    y_test = test_loader.dataset.test_labels.to(device)

    if binary:
        X_train = (X_train > 0.5).float()
        X_test = (X_test > 0.5).float()
    
    return train_loader, (X_test, y_test)


def log_path(args: argparse.Namespace):
    return args.log_path + '/' + args.name


def train(args: argparse.Namespace, model: DART):
    device = torch.device(args.device)
    model.to(device)
    
    train_loader, (X_test, y_test) = get_mnist_data(args.device, args.batch_size, args.binary)
    def random_test_sample(n: int = 100):
        return X_test[torch.randperm(X_test.shape[0])[:n]].view(n,-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not os.path.exists(log_path(args)):
        os.makedirs(log_path(args))

    writer = SummaryWriter(log_path(args))
        
    for epoch in range(args.epochs):
        for t, (X, _) in enumerate(train_loader):
            X = X.view(X.shape[0], -1).to(device)
            step = epoch * (len(train_loader.dataset) / args.batch_size) + t
            log_px, matrices = model.log_prob(X)

            print(f'Epoch: {epoch} Step: {step} log_px: {log_px.mean().item()}\r', end='')

            (-log_px).mean().backward()
            optimizer.step()
            optimizer.zero_grad()
            
            writer.add_scalar('Train_Likelihood', log_px.mean().item(), step)
            if t % args.interval == 0:
                test_data = random_test_sample().to(device)
                train_log_px, _ = model.log_prob(X, pause=args.pause)
                test_log_px, _ = model.log_prob(test_data, pause=args.pause)

                sample = model.sample(args.n_sample, device=device)
                dim = int(sample.shape[-1] ** 0.5)
                sample = sample.view(-1, 1, dim, dim)

                writer.add_scalar('Test_Likelihood', test_log_px.mean().item(), step)
                writer.add_images('Samples', sample, step)

                torch.save({
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict()
                }, f'{log_path(args)}/archive.pt')
                torch.save({
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict()
                }, f'{log_path(args)}/archive.pt')


def run(args: argparse.Namespace):
    seed = args.seed if args.seed is not None else 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dart = DART(784, args.hidden_size, args.n_hidden, args.alpha_dim, args.binary)

    train(args, dart)

if __name__ == '__main__':
    run(get_args())
