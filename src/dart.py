from typing import Optional, Union, List, Tuple
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from src.utils import fast_logexpmm, stable_logexpmm, dc_reduce, fast_logexpmv, fast_logexpmv_right


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
    def __init__(self, input_size: int, hidden_size: int, n_hidden: int, alpha_dim: int = 1, distribution: str = 'binary', **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.alpha_dim = alpha_dim
        self.distribution = distribution
        
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
        elif distribution == 'gaussian':
            self.distribution_param_count = 2

        output_values_per_dim = self.distribution_param_count * alpha_dim ** 2

        # last layer doesn't have nonlinear activation
        self.net += [MaskedLinear(self.hidden_size,
                                  self.input_size * output_values_per_dim,
                                  masks[-1].repeat(1,output_values_per_dim).view(-1,self.hidden_size)
        )]

        #p_alpha_eps = 1e-6
        #p_alpha = torch.ones(1, self.input_size - 1, 1, alpha_dim) * p_alpha_eps
        #p_alpha[:,:,:,0] = 1 - p_alpha_eps * (alpha_dim - 1)
        #p_alpha = torch.ones(1, self.input_size - 1, 1, alpha_dim) / float(alpha_dim)
        #self.unnormalized_log_p_alpha = torch.nn.Parameter(p_alpha.log())
        log_p_alpha = torch.empty(1, self.input_size - 1, 1, alpha_dim).normal_()
        self.unnormalized_log_p_alpha = torch.nn.Parameter(log_p_alpha)
        #self.unnormalized_log_p_alpha = p_alpha.log()
        #self.unnormalized_log_p_alpha.requires_grad = False
        
        self.net = nn.Sequential(*self.net)

    '''
    def to(self, *args, **kwargs):
        super().to(*args, *kwargs)
        self.unnormalized_log_p_alpha = self.unnormalized_log_p_alpha.to(*args, *kwargs)
    '''

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

    def forward(self, x, pause: bool = False):
        return self.log_prob(x, pause)

    def _sigmoid_gaussian(self, loc: torch.tensor, scale: torch.tensor):
        distribution = D.Normal(loc, scale)
        transform = D.transforms.SigmoidTransform()
        return D.TransformedDistribution(distribution, transform)

    def sample_alphas(self, n: int, device: str = 'cpu'):
        # Sample random latent "alpha" values
        # alpha_0 and alpha_n are fixed to have only one possible value, so we hard-code their indices to be zero here
        # the rest are sampled from our learned mixture prior defined by the logits returned by self._log_p_alpha()
        # we use these indices in the inner loop to pick which conditional to use at each step
        alpha_distribution = D.Categorical(logits=self._log_p_alpha().view(self.input_size - 1, self.alpha_dim))
        zeros = torch.zeros(n,1).long().to(device)
        alphas = torch.cat((zeros, alpha_distribution.sample((n,)), zeros), 1)
        return alphas

    def sample(self, n: int, device: str = 'cpu'):
        """
        Sample n samples from the model.
        """
        with torch.no_grad():
            z = torch.empty(n, self.input_size).to(device).normal_()
            x = torch.zeros_like(z)

            alphas = self.sample_alphas(n, device)
            for idx in range(self.input_size):
                alpha_row, alpha_column = alphas[:,idx], alphas[:,idx+1]
                theta = self.net(x).view(-1, self.input_size, self.distribution_param_count, *(self.alpha_dim,) * 2)
                batch_idx = torch.arange(theta.shape[0]).to(theta.device)

                if self.distribution == 'binary':
                    beta_logits = theta[batch_idx,idx,0,alpha_row,alpha_column]
                    d = D.Bernoulli(logits=beta_logits)
                    x[:,idx] = d.sample()
                elif self.distribution == 'categorical':
                    category_logits = theta[batch_idx,idx,:,alpha_row,alpha_column]
                    d = D.Categorical(logits=category_logits)
                    x[:,idx] = d.sample()
                elif self.distribution == 'gaussian':
                    mu = theta[batch_idx,idx,0,alpha_row,alpha_column]
                    std = theta[batch_idx,idx,1,alpha_row,alpha_column].exp()
                    d = self._sigmoid_gaussian(mu, std)
                    x[:,idx] = d.sample()

            assert (x >= 0).all()
            assert (x <= 1).all()

        return x, theta

    def log_prob(self, x, pause: bool = False):
        """
        Evaluate the log likelihood of a batch of samples.
        """
        if pause:
            import pdb; pdb.set_trace()

        theta = self.net(x).view(-1, self.input_size, self.distribution_param_count, *(self.alpha_dim,) * 2)
        if self.distribution == 'binary':
            d = D.Bernoulli(logits=theta[:,:,0])
        elif self.distribution == 'categorical':
            d = D.Categorical(logits=theta.permute(0,1,3,4,2)) # We need to permute to put the categories on the last dimension for D.Categorical
        elif self.distribution == 'gaussian':
            mu, std = theta[:,:,0], theta[:,:,1].exp()
            d = self._sigmoid_gaussian(mu, std)

        # Conditional probability matrices log p(x_i | alpha_{i-1}, alpha_i, x_{<i})
        log_px_matrices = d.log_prob(x.unsqueeze(-1).unsqueeze(-1))

        if self.alpha_dim > 1:
            # Multiply with the alpha marginals so that matrix multiplication corresponds to summing out alpha
            '''
            # For 'fold right', if that's your thing
            joint_matrices = log_px_matrices[:,1:] + self._log_p_alpha().transpose(-1,-2)

            # Reduce all of the "inner" matrices (all except x_1 and x_n)
            inner_matrix_product = reduce(fast_logexpmv_right, joint_matrices.transpose(0,1).flip(0))
            p_x1_g_alpha1 = log_px_matrices[:,0,0:1]
            log_px = fast_logexpmv_right(inner_matrix_product, p_x1_g_alpha1)
            '''
            joint_matrices = log_px_matrices[:,:-1] + self._log_p_alpha()

            # Reduce all of the "inner" matrices (all except x_1 and x_n)
            inner_matrix_product = reduce(fast_logexpmv, joint_matrices.transpose(0,1))
            p_xn_g_alpha_n1 = log_px_matrices[:,-1,:,0:1]
            log_px = fast_logexpmv(inner_matrix_product, p_xn_g_alpha_n1)
        else:
            log_px = log_px_matrices.sum(1).squeeze(-1).squeeze(-1)

        return log_px, theta
