from typing import Optional, Union, List, Tuple
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from src.utils import stable_logexpmm, fast_logexpmm


class MaskedLinear(nn.Linear):
    """Masked linear layer for MADE: takes in mask as input and masks out connections in the linear layers."""
    def __init__(self, input_size, output_size, mask):
        super().__init__(input_size, output_size)
        self.register_buffer('mask', mask)
        #self.weight[:]=(2/(input_size + output_size))

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

class PermuteLayer(nn.Module):
    """Layer to permute the ordering of inputs.

    Because our data is 2-D, forward() and inverse() will reorder the data in the same way.
    """
    def __init__(self, num_inputs):
        super(PermuteLayer, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])

    def forward(self, inputs):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device)

    def inverse(self, inputs):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device)


class DART(nn.Module):
    """DART based on Masked Autoencoder for Distribution Estimation.
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
            self.hidden_size, self.input_size * 2 * alpha_dim ** 2,
            masks[-1].repeat(1,2 * alpha_dim ** 2).view(-1,self.hidden_size))]

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

        theta = self.net(x).view(-1, self.input_size, 2, *(self.alpha_dim,) * 2)
        mu, std = theta[:,:,0], theta[:,:,1].exp()
        d = D.Normal(mu, std)
        log_p_alpha = torch.tensor(1./self.alpha_dim).float().log().to(mu.device)
        log_px_matrices = d.log_prob(x.unsqueeze(-1).unsqueeze(-1)) + log_p_alpha

        inner_matrices = list(log_px_matrices[:,1:-1].transpose(0,1))
        if len(inner_matrices) > 1:
            inner_matrix_product = reduce(fast_logexpmm, inner_matrices)
        else:
            inner_matrix_product = inner_matrices[0]

        first = log_px_matrices[:,0,0:1]
        last = log_px_matrices[:,-1,:,0:1]
        log_px = fast_logexpmm(first, fast_logexpmm(inner_matrix_product, last))

        return log_px, (mu, std)


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation.
    https://arxiv.org/abs/1502.03509

    Uses sequential ordering as in the MAF paper.
    Gaussian MADE to work with real-valued inputs"""
    def __init__(self, input_size, hidden_size, n_hidden):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        
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
            self.hidden_size, self.input_size * 2, masks[-1].repeat(2,1))]
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

    def forward(self, z):
        """
        Run the forward mapping (z -> x) for MAF through one MADE block.
        :param z: Input noise of size (batch_size, self.input_size)
        :return: (x, log_det). log_det should be 1-D (batch_dim,)
        """
        x = torch.zeros_like(z)
    
        # YOUR CODE STARTS HERE
        for idx in range(self.input_size):
            theta = self.net(x)
            mu, std = theta[:,idx], theta[:,self.input_size + idx].exp()
            x[:,idx] = z[:,idx] * std + mu
        log_det = None
        # YOUR CODE ENDS HERE

        return x, log_det

    def inverse(self, x):
        """
        Run one inverse mapping (x -> z) for MAF through one MADE block.
        :param x: Input data of size (batch_size, self.input_size)
        :return: (z, log_det). log_det should be 1-D (batch_dim,)
        """
        # YOUR CODE STARTS HERE
        theta = self.net(x)
        mu, alpha = theta[:,:self.input_size], theta[:,self.input_size:]
        z = (x - mu) / alpha.exp()
        log_det = -alpha.sum(-1)
        # YOUR CODE ENDS HERE

        return z, log_det


class MAF(nn.Module):
    """
    Masked Autoregressive Flow, using MADE layers.
    https://arxiv.org/abs/1705.07057
    """
    def __init__(self, input_size, hidden_size, n_hidden, n_flows):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.n_flows = n_flows
        self.base_dist = torch.distributions.normal.Normal(0,1)
        
        # need to flip ordering of inputs for every layer
        nf_blocks = []
        for i in range(self.n_flows):
            nf_blocks.append(
                MADE(self.input_size, self.hidden_size, self.n_hidden))
            nf_blocks.append(PermuteLayer(self.input_size))     # permute dims
        self.nf = nn.Sequential(*nf_blocks)

    def log_probs(self, x):
        """
        Obtain log-likelihood p(x) through one pass of MADE
        :param x: Input data of size (batch_size, self.input_size)
        :return: log_prob. This should be a Python scalar.
        """
        # YOUR CODE STARTS HERE
        z = x
        log_det_sum = 0
        for idx, flow in enumerate(self.nf):
            z, log_det = flow.inverse(z)
            log_det_sum += log_det.squeeze()
        log_prob = (self.base_dist.log_prob(z).sum(-1) + log_det_sum).mean()
        # YOUR CODE ENDS HERE

        return log_prob

    def loss(self, x):
        """
        Compute the loss.
        :param x: Input data of size (batch_size, self.input_size)
        :return: loss. This should be a Python scalar.
        """
        return -self.log_probs(x)

    def sample(self, device, n):
        """
        Draw <n> number of samples from the model.
        :param device: [cpu,cuda]
        :param n: Number of samples to be drawn.
        :return: x_sample. This should be a numpy array of size (n, self.input_size)
        """
        with torch.no_grad():
            x_sample = torch.randn(n, self.input_size).to(device)
            for flow in self.nf[::-1]:
                x_sample, log_det = flow.forward(x_sample)
            x_sample = x_sample.view(n, self.input_size)
            x_sample = x_sample.cpu().data.numpy()

        return x_sample


def test_grad():
    # run a quick and dirty test for the autoregressive property
    D = 2
    rng = np.random.RandomState(14)
    x = (rng.rand(1, D) > 0.5).astype(np.float32)

    configs = [
        (D, 4, 0, False),                 # test various hidden sizes
    ]

    for nin, hiddens, n_hidden, natural_ordering in configs:
        print("checking nin %d, hiddens %s, nout %d, natural %s" %
              (nin, hiddens, n_hidden, natural_ordering))
        model = DART(nin, hiddens, n_hidden)

        # run backpropagation for each dimension to compute what other
        # dimensions it depends on.
        res = []
        for k in range(nin):
            xtr = torch.from_numpy(x)
            xtr.requires_grad=True
            xtrhat = model(xtr)

            loss = xtrhat[0,k]
            loss.backward()

            import pdb; pdb.set_trace()
            depends = (xtr.grad[0].numpy() != 0).astype(np.uint8)
            depends_ix = list(np.where(depends)[0])
            isok = k % nin not in depends_ix

            res.append((len(depends_ix), k, depends_ix, isok))

            # pretty print the dependencies
            res.sort()

    for nl, k, ix, isok in res:
        print("output %2d depends on inputs: %30s : %s" % (k, ix, "OK" if isok else "NOTOK"))
            
            
def test():
    x_dim = 4
    d = DART(x_dim,2048,2,alpha_dim=1).to(torch.device('cuda'))
    #x = torch.empty(1000,x_dim).normal_().to(torch.device('cuda'))
    #x = torch.stack((torch.arange(3), torch.arange(3))).float().to(torch.device('cuda')) * 10000000
    x = torch.eye(x_dim).float().to(torch.device('cuda'))
    #x = torch.cat((x, torch.tensor([[1.,1.,0],[1.,1.,1.]]).to(torch.device('cuda'))))
    #x[x>0] = torch.tensor(0.).log().to(torch.device('cuda'))
    x[x>0] = 1e10
    print('start')
    logpx, matrices = d(x)
    print('sample')
    sample = d.sample(10, torch.device('cuda'))
    print(f'isnan: {torch.isnan(logpx).any()}')
    print(f'isinf: {torch.isinf(logpx).any()}')
    import pdb; pdb.set_trace()
    
if __name__ == '__main__':
    test_grad()
