from typing import Optional, Union, List, Tuple
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from src.utils import fast_logexpmv, dc_reduce


class TT(nn.Module):
    def __init__(self, input_size: int, alpha_dim: int = 1):
        super().__init__()

        self.input_size = input_size
        self.alpha_dim = alpha_dim
        
        log_p_alpha = torch.empty(1, self.input_size - 1, 1, alpha_dim).normal_()
        self.unnormalized_log_p_alpha = torch.nn.Parameter(log_p_alpha)

        self.theta = nn.Parameter(torch.empty((1, input_size, alpha_dim, alpha_dim)).uniform_())

    def _log_p_alpha(self):
        u_log_p = self.unnormalized_log_p_alpha
        max_u = u_log_p.max(-1, keepdim=True).values
        logsumexp = (u_log_p - max_u).exp().sum(-1, keepdim=True).log()
        log_p = u_log_p - (max_u + logsumexp)

        assert ((log_p.exp().sum(-1) - 1).abs() < 1e-6).all()
        return log_p

    def forward(self, x):
        return self.log_prob(x)

    def sample_alphas(self, n: int, device: str = 'cpu'):
        # Sample random latent "alpha" values
        # alpha_0 and alpha_n are fixed to have only one possible value, so we hard-code their indices to be zero here
        # the rest are sampled from our learned mixture prior defined by the logits returned by self._log_p_alpha()
        # we use these indices in the inner loop to pick which conditional to use at each step
        alpha_distribution = D.Categorical(logits=self._log_p_alpha().view(self.input_size - 1, self.alpha_dim))
        zeros = torch.zeros(n,1).long().to(device)
        alphas = torch.cat((zeros, alpha_distribution.sample((n,)), zeros), 1)
        return alphas

    def sample(self, n: int = 1, device: str = 'cpu'):
        alphas = self.sample_alphas(n, device)

        alpha_rows = alphas[:,:-1]
        alpha_columns = alphas[:,1:]

        batch_idx = torch.zeros(n,self.input_size).to(device).long()
        element_idx = torch.arange(self.input_size).to(device).view(1,-1).repeat(n,1)

        thetas = self.theta[batch_idx,element_idx,alpha_rows,alpha_columns]

        return D.Bernoulli(logits=thetas).sample(), self.theta
    
    def log_prob(self, x):
        # Conditional probability matrices log p(x_i | alpha_{i-1}, alpha_i, x_{<i})
        #d = D.Bernoulli(logits=self.theta)
        #log_px_matrices = d.log_prob(x.unsqueeze(-1).unsqueeze(-1))
        log_px_matrices = -self.theta.clamp(min=0) + self.theta * x.unsqueeze(-1).unsqueeze(-1) - (1 + (-self.theta.abs()).exp()).log()
        
        # Multiply with the alpha marginals so that matrix multiplication corresponds to summing out alpha
        joint_matrices = log_px_matrices[:,:-1] + self._log_p_alpha()

        # Reduce all of the "inner" matrices (all except x_1 and x_n)
        left_reduce_product = reduce(fast_logexpmv, joint_matrices.transpose(0,1))
        p_xn_g_alpha_n1 = log_px_matrices[:,-1,:,0:1]
        log_px = fast_logexpmv(left_reduce_product, p_xn_g_alpha_n1)

        return log_px, self.theta
