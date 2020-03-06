from typing import Optional, Union, List, Tuple
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from src.utils import fast_logexpmv, fast_logexpmm, stable_logexpmm, dc_reduce


class Tree(nn.Module):
    def __init__(self, input_size: int, alpha_dim: int = 1):
        pass


class TT(nn.Module):
    def __init__(self, input_size: int, alpha_dim: int = 1):
        super().__init__()

        self.input_size = input_size
        self.alpha_dim = alpha_dim
        
        self.u_log_p_a1 = nn.Parameter(torch.empty(1, 1, 1, alpha_dim).normal_())
        self.u_log_transition_matrices = nn.Parameter(torch.empty(self.input_size - 1, alpha_dim, alpha_dim).normal_())
        self.log_p_xi_g_ai = nn.Parameter(torch.empty(1, self.input_size, 1, alpha_dim).normal_())

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

    def sample_alphas(self, n: int, device: str = 'cpu'):
        # Sample random latent "alpha" values
        # alpha_0 and alpha_n are fixed to have only one possible value, so we hard-code their indices to be zero here
        # the rest are sampled from our learned mixture prior defined by the logits returned by self._log_p_alpha()
        # we use these indices in the inner loop to pick which conditional to use at each step
        alphas = []
        logits = self.log_p_a1.repeat((n,1,1,1))
        for idx in range(self.input_size):
            alpha_d = D.Categorical(logits=logits.squeeze())
            alphas.append(alpha_d.sample())
            if idx < self.input_size - 1:
                logits = self.log_transition.squeeze()[idx,alphas[-1]]
        return torch.stack(alphas, 1)
        

    def sample(self, n: int = 1, device: str = 'cpu'):
        alphas = self.sample_alphas(n, device)
        betas = self.log_p_xi_g_ai.squeeze(-2).repeat(n,1,1).gather(-1, alphas.unsqueeze(-1))

        return D.Bernoulli(logits=betas.squeeze()).sample(), betas
    
    def log_prob(self, x):
        # Conditional probability matrices log p(x_i | alpha_{i-1}, alpha_i, x_{<i})
        d = D.Bernoulli(logits=self.log_p_xi_g_ai)
        log_px_vectors = d.log_prob(x.unsqueeze(-1).unsqueeze(-1))
        
        # Multiply with the alpha marginals so that matrix multiplication corresponds to summing out alpha
        #log_px_vectors = torch.cat((log_px_vectors[:,0:1] + self.log_p_a1, log_px_vectors[:,1:]), 1)
        
        reduction_matrices = log_px_vectors[:,1:] + self.log_transition
        log_pxa1 = log_px_vectors[:,0:1] + self.log_p_a1
        reduction_matrices = torch.cat((log_pxa1.repeat(1,1,self.alpha_dim,1), reduction_matrices), 1)
        
        # Reduce all of the "inner" matrices (all except x_1 and x_n)
        #left_reduce_product = reduce(fast_logexpmm, reduction_matrices.transpose(0,1))
        #complete_product = fast_logexpmv(log_px_vectors[:,0], left_reduce_product)
        #log_px = complete_product.logsumexp(-1)
        p_x_g_an = reduce(fast_logexpmv, reduction_matrices.transpose(0,1))
        log_px = p_x_g_an.logsumexp(-1)
        return log_px, self.log_p_a1
