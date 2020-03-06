import torch

from src.dart import DART
from src.tensor import TT

dim = 10

d = DART(dim, 10, 2, 1, 'binary')
d2 = DART(dim, 10, 2, 2, 'binary')
d4 = DART(dim, 10, 2, 4, 'binary')
d100 = DART(dim, 10, 2, 100, 'binary')

tt4 = TT(dim, 25)

#tt4.sample(10)

X = torch.zeros(2 ** dim, dim)
idx = torch.arange(2 ** dim)
for d_ in range(dim):
    X[(idx // 2 ** d_) % 2 == 0,d_] = 1

print(d(X)[0].exp().sum())
print(d2(X)[0].exp().sum())
print(d4(X)[0].exp().sum())
print(d100(X)[0].exp().sum())
print(tt4(X)[0].exp().sum())

samples = tt4.sample(1000000)
for x in X:
    p_sample = (samples[0] == x).all(-1).float().mean()
    p_model = tt4(x.unsqueeze(0))[0].exp().squeeze()
    print(p_sample.item(), p_model.item(), ((p_sample - p_model) / p_model).item())
