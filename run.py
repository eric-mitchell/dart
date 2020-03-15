import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import argparse
import os
import random
import time


from src.args import get_args
from src.dart import DART, DARTHMM
from src.tensor import TT, MPS
from src.data import get_data


def log_path(args: argparse.Namespace):
    return args.log_path + '/' + args.name


def train(args: argparse.Namespace, model: DART):
    device = torch.device(args.device)
    model.to(device)

    train_loader, test_loader = get_data(args.dataset, args.distribution,
                                         args.batch_size * (5 if args.filter_class is not None else 1),
                                         args.test_batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, 3, gamma=1)
    
    if not os.path.exists(log_path(args)):
        os.makedirs(log_path(args))

    writer = SummaryWriter(log_path(args))

    for epoch in range(args.epochs):
        for t, (X, y) in enumerate(train_loader):
            model.train()
            if t == 1:
                start = time.time()

            if args.profile and t > 50:
                break

            if args.filter_class is not None:
                X = X[y==args.filter_class]
                if len(X) == 0:
                    continue

            X = X.view(X.shape[0], -1).to(device)
            step = int(epoch * (len(train_loader.dataset) / args.batch_size)) + t
            log_px, matrices = model(X)

            loss = -log_px.mean()
            loss.mean().backward()

            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            avg_time = 0 if t == 0 else (time.time() - start) / t
            print(f'Epoch: {epoch}\tStep: {step}\tlog_px: {log_px.mean().item():#.6g}\t{avg_time:#.6g}\r', end='')

            writer.add_scalar('Grad', grad, step)
            writer.add_scalar('Train_Likelihood', log_px.mean().item(), step)

            if step % args.vis_interval == 0:
                model.eval()
                # Grab just one random test data batch
                for test_data, _ in test_loader:
                    test_data = test_data.view(test_data.shape[0], -1).to(device)
                    break

                with torch.no_grad():
                    train_log_px, train_matrices = model(X[:args.n_sample])
                    test_log_px, test_matrices = model(test_data[:args.n_sample])

                sample, theta = model.sample(args.n_sample, device)
                dim = int(sample.shape[-1] ** 0.5)
                sample = sample.view(-1, 1, dim, dim)
                if args.distribution == 'gaussian':
                    sample = sample.sigmoid()

                writer.add_scalar('Test_Likelihood', test_log_px.mean().item(), step)
                writer.add_images('Samples', sample, step)

                if args.distribution == 'gaussian':
                    alphas = model.sample_alphas(args.n_sample, device)
                    batch_idx = torch.arange(train_matrices.shape[0], device=train_matrices.device)
                    
                    mus, log_sigmas = [], []
                    theta_idx = torch.arange(theta.shape[1], device=theta.device)
                    for theta_, alpha_ in zip(theta, alphas):
                        mus.append(theta_[theta_idx, 0, alpha_[:-1], alpha_[1:]].view(1, dim, dim).sigmoid())
                        log_sigmas.append(theta_[theta_idx, 1, alpha_[:-1], alpha_[1:]].view(1, dim, dim))

                    writer.add_images('Sample_mus', torch.stack(mus))
                    writer.add_images('Sample_sigmas', torch.stack(log_sigmas).exp())

            if step % args.save_interval == 0:
                if not args.no_save:
                    torch.save({
                        'model': model.state_dict(),
                        'opt': optimizer.state_dict()
                    }, f'{log_path(args)}/archive.pt')

        scheduler.step()


def run(args: argparse.Namespace):
    seed = args.seed if args.seed is not None else 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Extra parameters to the DART model constructor
    kwargs = {}
    if args.distribution == 'categorical':
        kwargs['categories'] = 256

    if args.model == 'dart':
        model = DART(784, args.hidden_size, args.n_hidden, args.alpha_dim, args.distribution, dropout=args.dropout, **kwargs)
    if args.model == 'darthmm':
        model = DARTHMM(784, args.hidden_size, args.n_hidden, args.alpha_dim, args.distribution, dropout=args.dropout, **kwargs)
    elif args.model == 'tt':
        assert args.distribution == 'binary', 'TT model only works with binary variables'
        model = TT(784, args.alpha_dim)
    elif args.model == 'mps':
        model = MPS(784, args.alpha_dim, 2)

    if args.archive is not None:
        model.load_state_dict(torch.load(args.archive)['model'])

    if args.pdb:
        import pdb; pdb.set_trace()

    if args.profile:
        args.epochs = 1
        
    with torch.autograd.profiler.profile(enabled=args.profile, use_cuda=True) as prof:
        train(args, model)

    if args.profile:
        with open(f'{args.name}.results', 'w') as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total"))
        import pdb; pdb.set_trace()

if __name__ == '__main__':
    run(get_args())
