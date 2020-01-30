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

        self.distribution_param_count = 1 if binary else 2 # 1 for Bernoulli, 2 for Gaussian
        output_values_per_dim = self.distribution_param_count * alpha_dim ** 2
        self.net += [MaskedLinear(self.hidden_size,
                                  self.input_size * output_values_per_dim,
                                  masks[-1].repeat(1,output_values_per_dim).view(-1,self.hidden_size)
        )]

        log_p_alpha = torch.ones(1, self.input_size - 1, 1, alpha_dim) / float(alpha_dim)
        self.unnormalized_log_p_alpha = torch.nn.Parameter(log_p_alpha.log())

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

    def _log_p_alpha(self):
        u_log_p = self.unnormalized_log_p_alpha
        max_u = u_log_p.max(-1, keepdim=True).values
        logsumexp = (u_log_p - max_u).exp().sum(-1, keepdim=True).log()
        log_p = u_log_p - (max_u + logsumexp)

        assert ((log_p.exp().sum(-1) - 1).abs() < 1e-6).all()
        return log_p

    def forward(self, x):
        return self.log_prob(x)

    def sample(self, n: int, device: str = 'cpu'):
        """
        Sample n samples from the model.
        """
        with torch.no_grad():
            z = torch.empty(n, self.input_size).to(device).normal_()
            x = torch.zeros_like(z)

            alpha_distribution = D.Categorical(logits=self._log_p_alpha().view(self.input_size - 1, self.alpha_dim))
            zeros = torch.zeros(n,1).long().to(device)
            alphas = torch.cat((zeros, alpha_distribution.sample((n,)), zeros), 1)

            for idx in range(self.input_size):
                alpha_row, alpha_column = alphas[:,idx], alphas[:,idx+1]
                if self.binary:
                    theta = self.net(x).view(-1, self.input_size, *(self.alpha_dim,) * 2)
                else:
                    theta = self.net(x).view(-1, self.input_size, 2, *(self.alpha_dim,) * 2)

                batch_idx = torch.arange(theta.shape[0]).to(theta.device)
                if self.binary:
                    beta_logits = theta[batch_idx,idx,alpha_row,alpha_column]
                    d = D.Bernoulli(logits=beta_logits)
                    x[:,idx] = d.sample()
                else:
                    mu = theta[batch_idx,idx,0,alpha_row,alpha_column]
                    std = theta[batch_idx,idx,1,alpha_row,alpha_column].exp() + 1e-2
                    d = D.Normal(mu, std)
                    #x[:,idx] = z[:,idx] * std + mu
                    x[:,idx] = d.sample()

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
            mu, std = theta[:,:,0], theta[:,:,1].exp() + 1e-2
            d = D.Normal(mu, std)

        log_px_matrices = d.log_prob(x.unsqueeze(-1).unsqueeze(-1))
        joint_matrices = log_px_matrices[:,:-1] + self._log_p_alpha()

        inner_matrices = list(joint_matrices[:,1:].transpose(0,1))
        if len(inner_matrices) > 1:
            inner_matrix_product = reduce(stable_logexpmm, inner_matrices)
        else:
            inner_matrix_product = inner_matrices[0]

        first = joint_matrices[:,0,0:1]
        last = log_px_matrices[:,-1,:,0:1]
        log_px = stable_logexpmm(first, stable_logexpmm(inner_matrix_product, last))

        return log_px, theta
