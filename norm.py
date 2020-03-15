import torch

from src.dart import DART, DARTHMM
from src.tensor import TT, MPS

dim = 7
dev = 'cpu'
d = DART(dim, 10, 2, 1, 'binary').to(dev)
d2 = DART(dim, 10, 2, 2, 'binary').to(dev)
d4hmm = DARTHMM(dim, 10, 2, 16, 'binary').to(dev)
d100 = DART(dim, 10, 2, 8, 'binary').to(dev)

tt4 = TT(dim, 9).to(dev)
mps4 = MPS(dim, 9, 2).to(dev)

#tt4.sample(10)

X = torch.zeros(2 ** dim, dim).to(dev)
idx = torch.arange(2 ** dim).to(dev)
for d_ in range(dim):
    X[(idx // 2 ** d_) % 2 == 0,d_] = 1

print(d(X)[0].exp().sum())
print(d2(X)[0].exp().sum())
print(d4hmm(X)[0].exp().sum())
print(d100(X)[0].exp().sum())
print(tt4(X)[0].exp().sum())
print(mps4(X)[0].exp().sum())

sample_model = d4hmm
samples = sample_model.sample(1000000, dev)
for x in X:
    p_sample = (samples[0] == x).all(-1).float().mean()
    p_model = sample_model(x.unsqueeze(0))[0].exp().squeeze()
    print(p_sample.item(), p_model.item(), ((p_sample - p_model) / p_model).item())
