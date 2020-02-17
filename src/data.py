import torch
from torchvision import datasets, transforms


def get_mnist_data(device, distribution: str, train_batch_size: int, test_batch_size: int):
    preprocess = transforms.ToTensor()
    if distribution == 'binary':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x > 0.5,
            lambda x: x.float()
        ])
    elif distribution == 'categorical':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x * 255
        ])
    elif distribution == 'gaussian':
        noise = 0.05
        eps = 1e-3
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x * (1 - noise - eps) + (noise + eps) / 2, # Squash slightly
            lambda x: x + torch.empty_like(x).uniform_(-eps/2,eps/2) # Add noise
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=preprocess),
        batch_size=train_batch_size,  # Using a weird batch size to prevent students from hard-coding
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=preprocess),
        batch_size=test_batch_size,
        shuffle=True)

    return train_loader, test_loader
