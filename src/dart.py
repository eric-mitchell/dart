from typing import Optional, Union, List, Tuple
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from src.utils import fast_logexpmv


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
    def __init__(self, input_size: int, hidden_size: int, n_hidden: int, alpha_dim: int = 1, distribution: str = 'binary', sampling_order: str = 'sequential', **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.alpha_dim = alpha_dim
        self.distribution = distribution
        self.sampling_order = sampling_order
        
        masks = self.create_masks()

        # construct layers: inner, hidden(s), output
        self.net = [MaskedLinear(self.input_size, self.hidden_size, masks[0])]
        self.net += [nn.ReLU(inplace=True)]
        # iterate over number of hidden layers
        for i in range(self.n_hidden - 1):
            self.net += [MaskedLinear(
                self.hidden_size, self.hidden_size, masks[i+1])]
            self.net += [nn.ReLU(inplace=True)]

        if distribution == 'binary':
            self.distribution_param_count = 1
        elif distribution == 'categorical':
            if 'categories' not in kwargs:
                raise ValueError('Must pass number of categories as categories=n_categories')
            self.distribution_param_count = kwargs['categories']
            self.kwargs = kwargs
        elif distribution == 'gaussian':
            self.distribution_param_count = 2

        output_values_per_dim = self.distribution_param_count * alpha_dim

        # last layer doesn't have nonlinear activation
        self.net += [MaskedLinear(self.hidden_size,
                                  self.input_size * output_values_per_dim,
                                  masks[-1].repeat(1,output_values_per_dim).view(-1,self.hidden_size)
        )]

        self.u_log_p_a1 = nn.Parameter(torch.empty(1, 1, 1, alpha_dim).normal_())
        self.u_log_transition_matrices = nn.Parameter(torch.empty(self.input_size - 1, alpha_dim, alpha_dim).normal_())
        
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

    def _normalize_logits(self, unnorm: torch.tensor, dim: int = -1):
        max_u = unnorm.max(dim, keepdim=True).values
        logsumexp = (unnorm - max_u).exp().sum(dim, keepdim=True).log()
        norm = unnorm - (max_u + logsumexp)

        assert ((norm.exp().sum(dim) - 1).abs() < 1e-5).all()
        return norm
        
    @property
    def log_p_a1(self):
        return self._normalize_logits(self.u_log_p_a1)

    @property
    def log_transition(self):
        return self._normalize_logits(self.u_log_transition_matrices, dim=-1)

    def forward(self, x):
        return self.log_prob(x)

    def _sigmoid_gaussian(self, loc: torch.tensor, scale: torch.tensor):
        distribution = D.Normal(loc, scale)
        transform = D.transforms.SigmoidTransform()
        return D.TransformedDistribution(distribution, transform)

    def sample_alphas(self, n: int, device: str = 'cpu', a1: int = None):
        alphas = []
        logits = self.log_p_a1.repeat((n,1,1,1))
        if a1 is not None:
            logits[:,:,:,a1] = 100
            if a1 > 0:
                logits[:,:,:,:a1] = 0
            if a1 < self.alpha_dim - 1:
                logits[:,:,:,a1+1:] = 0
            
        for idx in range(self.input_size):
            alpha_d = D.Categorical(logits=logits.squeeze())
            alphas.append(alpha_d.sample())
            if idx < self.input_size - 1:
                logits = self.log_transition.squeeze()[idx,alphas[-1]]
        return torch.stack(alphas, 1)

    def sample(self, n: int, device: str = 'cpu', a1: int = None):
        """
        Sample n samples from the model.
        """
        with torch.no_grad():
            x = torch.zeros(n, self.input_size).to(device)

            alphas = self.sample_alphas(n, device, a1)
            for idx in range(self.input_size):
                theta = self.net(x).view(-1, self.input_size, self.distribution_param_count, self.alpha_dim)
                alpha = alphas[:,idx]
                batch_idx = torch.arange(theta.shape[0]).to(theta.device)

                if self.distribution == 'binary':
                    beta_logits = theta[batch_idx,idx,0,alpha]
                    d = D.Bernoulli(logits=beta_logits)
                    x[:,idx] = d.sample()
                elif self.distribution == 'categorical':
                    category_logits = theta[batch_idx,idx,:,alpha]
                    d = D.Categorical(logits=category_logits)
                    x[:,idx] = d.sample().float() / (self.kwargs['categories'] - 1)
                elif self.distribution == 'gaussian':
                    mu = theta[batch_idx,idx,0,alpha]
                    std = theta[batch_idx,idx,1,alpha].exp()
                    d = self._sigmoid_gaussian(mu, std)
                    x[:,idx] = d.sample()

            assert (x >= 0).all()
            assert (x <= 1).all()

        return x, theta

    def log_prob(self, x):
        """
        Evaluate the log likelihood of a batch of samples.
        """
        theta = self.net(x).view(-1, self.input_size, self.distribution_param_count, self.alpha_dim)
        if self.distribution == 'binary':
            d = D.Bernoulli(logits=theta)
        elif self.distribution == 'categorical':
            d = D.Categorical(logits=theta.permute(0,1,3,2)) # We need to permute to put the categories on the last dimension for D.Categorical
        elif self.distribution == 'gaussian':
            mu, std = theta[:,:,0], theta[:,:,1].exp()
            d = self._sigmoid_gaussian(mu, std)

        # We need the indices for categorical distribution
        if self.distribution == 'categorical':
            x = x * (self.kwargs['categories'] - 1)

        # Conditional probability matrices log p(x_i | alpha_{i-1}, alpha_i, x_{<i})
        log_px_vectors = d.log_prob(x.unsqueeze(-1).unsqueeze(-1))

        if self.alpha_dim > 1 or True:
            # Multiply with the alpha marginals so that matrix multiplication corresponds to summing out alpha
            reduction_matrices = log_px_vectors[:,1:] + self.log_transition
            log_pxa1 = log_px_vectors[:,0:1] + self.log_p_a1
            reduction_matrices = torch.cat((log_pxa1.repeat(1,1,self.alpha_dim,1), reduction_matrices), 1)
            
            # Reduce all of the "inner" matrices (all except x_1 and x_n)
            p_x_g_an = reduce(fast_logexpmv, reduction_matrices.transpose(0,1))
            log_px = p_x_g_an.logsumexp(-1)
        else:
            log_px = log_px_matrices.sum(1).squeeze(-1).squeeze(-1)

        return log_px, theta
