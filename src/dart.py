from typing import Optional, Union, List, Tuple
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from src.utils import stable_logexpmm


class MaskedLinear(nn.Linear):
    """Masked linear layer for MADE: takes in mask as input and masks out connections in the linear layers."""
    def __init__(self, input_size, output_size, mask):
        super().__init__(input_size, output_size)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class BDART(nn.Module):
    """BDART based on Masked Autoencoder for Distribution Estimation.
    Gaussian MADE to work with real-valued inputs"""
    def __init__(self, input_size: int, hidden_size: int, n_hidden: int, alpha_dim: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.alpha_dim = alpha_dim

        masks = self.create_masks()

        # construct layers: inner, hidden(s), output
        self.net = [MaskedLinear(self.input_size, self.hidden_size, masks[0])]
        self.net += [nn.ReLU(inplace=True)]
        # iterate over number of hidden layers
        for i in range(self.n_hidden):
            self.net += [MaskedLinear(
                self.hidden_size, self.hidden_size, masks[i+1])]
            self.net += [nn.ReLU(inplace=True)]
        # last layer doesn't have nonlinear activation
        self.net += [MaskedLinear(
            self.hidden_size, self.input_size * alpha_dim ** 2,
            masks[-1].repeat(1,alpha_dim ** 2).view(-1,self.hidden_size))]

        self.net = nn.Sequential(*self.net)

    def create_masks(self):
        """
        Creates masks for sequential (natural) ordering.
        """
        masks = []
        input_degrees = torch.arange(self.input_size)
        degrees = [input_degrees] # corresponds to m(k) in paper

        # iterate through every hidden layer
        for n_h in range(self.n_hidden+1):
            degrees += [torch.arange(self.hidden_size) % (self.input_size - 1)]
        degrees += [input_degrees % self.input_size - 1]

        self.m = degrees
        # output layer mask
        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

        return masks

    def forward(self, x):
        return self.log_prob(x)
    
    def sample(self, n: int, device: str = 'cpu'):
        """
        Sample n samples from the model.
        """
        simple_alpha = True
        with torch.no_grad():
            z = torch.empty(n, self.input_size).to(device).normal_()
            x = torch.zeros_like(z)

            for idx in range(self.input_size):
                theta = self.net(x).view(-1, self.input_size, *(self.alpha_dim,) * 2)
                if simple_alpha:
                    alphas = [0] * 2
                else:
                    for dim_idx, dim in enumerate(theta.shape[3:]):
                        if (idx == 0 and dim_idx == 0) or (idx == self.input_size - 1 and dim_idx == 0):
                            alphas.append(torch.zeros(theta.shape[0]).long().to(device))
                        else:
                            alphas.append(D.Categorical(torch.ones(dim)/dim).sample((theta.shape[0],)).to(device))

                batch_idx = torch.arange(theta.shape[0]).to(theta.device)
                beta_logits = theta[batch_idx,idx,alphas[0],alphas[1]]
                d = D.Bernoulli(logits=beta_logits)
                x[:,idx] = d.sample()

        return x

    def log_prob(self, x, pause: bool = False):
        """
        Evaluate the log likelihood of a batch of samples.
        """
        if pause:
            import pdb; pdb.set_trace()

        beta_logits = self.net(x).view(-1, self.input_size, 2, *(self.alpha_dim,) * 2)
        d = D.Bernoulli(logits=beta_logits)
        log_p_alpha = torch.tensor(1./self.alpha_dim).float().log().to(mu.device)
        log_px_matrices = d.log_prob(x.unsqueeze(-1).unsqueeze(-1)) + log_p_alpha

        inner_matrices = list(log_px_matrices[:,1:-1].transpose(0,1))
        if len(inner_matrices) > 1:
            inner_matrix_product = reduce(stable_logexpmm, inner_matrices)
        else:
            inner_matrix_product = inner_matrices[0]

        first = log_px_matrices[:,0,0:1]
        last = log_px_matrices[:,-1,:,0:1]
        log_px = stable_logexpmm(first, stable_logexpmm(inner_matrix_product, last))

        return log_px, (mu, std)

    
class DART(nn.Module):
    """DART based on Masked Autoencoder for Distribution Estimation.
    Gaussian MADE to work with real-valued inputs"""
    def __init__(self, input_size: int, hidden_size: int, n_hidden: int, alpha_dim: int = 1, binary: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.alpha_dim = alpha_dim
        self.binary = binary
        
        masks = self.create_masks()

        # construct layers: inner, hidden(s), output
        self.net = [MaskedLinear(self.input_size, self.hidden_size, masks[0])]
        self.net += [nn.ReLU(inplace=True)]
        # iterate over number of hidden layers
        for i in range(self.n_hidden):
            self.net += [MaskedLinear(
                self.hidden_size, self.hidden_size, masks[i+1])]
            self.net += [nn.ReLU(inplace=True)]
        # last layer doesn't have nonlinear activation
        distribution_param_count = 1 if binary else 2
        self.net += [MaskedLinear(
            self.hidden_size, self.input_size * distribution_param_count * alpha_dim ** 2,
            masks[-1].repeat(1,distribution_param_count * alpha_dim ** 2).view(-1,self.hidden_size))]

        self.net = nn.Sequential(*self.net)

    def create_masks(self):
        """
        Creates masks for sequential (natural) ordering.
        """
        masks = []
        input_degrees = torch.arange(self.input_size)
        degrees = [input_degrees] # corresponds to m(k) in paper

        # iterate through every hidden layer
        for n_h in range(self.n_hidden+1):
            degrees += [torch.arange(self.hidden_size) % (self.input_size - 1)]
        degrees += [input_degrees % self.input_size - 1]

        self.m = degrees
        # output layer mask
        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

        return masks

    def forward(self, x):
        return self.log_prob(x)
    
    def sample(self, n: int, device: str = 'cpu'):
        """
        Sample n samples from the model.
        """
        simple_alpha = True
        with torch.no_grad():
            z = torch.empty(n, self.input_size).to(device).normal_()
            x = torch.zeros_like(z)

            for idx in range(self.input_size):
                if self.binary:
                    theta = self.net(x).view(-1, self.input_size, *(self.alpha_dim,) * 2)
                else:
                    theta = self.net(x).view(-1, self.input_size, 2, *(self.alpha_dim,) * 2)
                if simple_alpha:
                    alphas = [0] * 2
                else:
                    for dim_idx, dim in enumerate(theta.shape[3:]):
                        if (idx == 0 and dim_idx == 0) or (idx == self.input_size - 1 and dim_idx == 0):
                            alphas.append(torch.zeros(theta.shape[0]).long().to(device))
                        else:
                            alphas.append(D.Categorical(torch.ones(dim)/dim).sample((theta.shape[0],)).to(device))

                batch_idx = torch.arange(theta.shape[0]).to(theta.device)
                if self.binary:
                    beta_logits = theta[batch_idx,idx,alphas[0],alphas[1]]
                    d = D.Bernoulli(logits=beta_logits)
                    x[:,idx] = d.sample()
                else:
                    mu = theta[batch_idx,idx,0,alphas[0],alphas[1]]
                    std = theta[batch_idx,idx,1,alphas[0],alphas[1]].exp()
                    x[:,idx] = z[:,idx] * std + mu

        return x

    def log_prob(self, x, pause: bool = False):
        """
        Evaluate the log likelihood of a batch of samples.
        """
        if pause:
            import pdb; pdb.set_trace()

        if self.binary:
            theta = self.net(x).view(-1, self.input_size, *(self.alpha_dim,) * 2)
            d = D.Bernoulli(logits=theta)
        else:
            theta = self.net(x).view(-1, self.input_size, 2, *(self.alpha_dim,) * 2)
            mu, std = theta[:,:,0], theta[:,:,1].exp()
            d = D.Normal(mu, std)

        log_p_alpha = torch.tensor(1./self.alpha_dim).float().log().to(theta.device)
        log_px_matrices = d.log_prob(x.unsqueeze(-1).unsqueeze(-1)) + log_p_alpha # we need the joint to marginalize, not conditional on alpha

        inner_matrices = list(log_px_matrices[:,1:-1].transpose(0,1))
        if len(inner_matrices) > 1:
            inner_matrix_product = reduce(stable_logexpmm, inner_matrices)
        else:
            inner_matrix_product = inner_matrices[0]

        first = log_px_matrices[:,0,0:1]
        last = log_px_matrices[:,-1,:,0:1]
        log_px = stable_logexpmm(first, stable_logexpmm(inner_matrix_product, last))

        return log_px, theta
