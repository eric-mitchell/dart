import argparse
import torch
import time
from typing import Callable, List
from functools import reduce
import torchviz


def dc_reduce(f: Callable, values: torch.tensor):
    n = len(values)
    if n == 1:
        return values[0]
    elif n == 2:
        return f(values[0], values[1])
    else:
        return f(dc_reduce(f, values[:n // 2]), dc_reduce(f, values[n // 2:]))


def l_reduce(f: Callable, values):
    return reduce(f, values)


def logexpmm(A, B, time_=False):
    start = time.time()
    result = (A.exp().bmm(B.exp())).log()

    return result, time.time() - start


def fast_logexpmm(A, B, time_=False, stable: bool = True):
    if time_:
        start = time.time()

    if stable:
        max_A = A.max(-1, keepdim=True).values
        max_B = B.max(-2, keepdim=True).values
    else:
        max_A = A.max(-1, keepdim=True).values.max(-2, keepdim=True).values
        max_B = B.max(-1, keepdim=True).values.max(-2, keepdim=True).values

    C = (A - max_A).exp().bmm((B - max_B).exp())

    if C.min() <= 0:
        print('Taking log of zero...')
        import pdb; pdb.set_trace()

    C = C.log()
    C += max_A + max_B
    if time_:
        return C, time.time() - start
    else:
        return C

def old_logexpmv(A, B, time_=False):
    # Left multiply a batch of row vectors A with a batch of matrices B
    assert B.shape[-2] == A.shape[-1]
    if time_:
        start = time.time()
    
    C = A.transpose(-1,-2) + B
    maxC = C.max(-2, keepdim=True).values
    C = (C - maxC).exp().sum(-2,keepdim=True).log() + maxC
    
    if time_:
        return C, time.time() - start
    else:
        return C


def fast_logexpmv(A, B, A_idx: int = 0):
    # Left multiply a batch of row vectors A with a batch of matrices B
    if A.shape[-2] != 1:
        # if A is a batch of matrices, trim to just take the first row
        A = A[:,A_idx:A_idx + 1]

    C = A.transpose(-1,-2) + B
    maxC = C.max(-2, keepdim=True).values
    return (C - maxC).exp().sum(-2,keepdim=True).log() + maxC


def fast_logexpmv_right(A, B, A_idx: int = 0):
    # Left multiply a batch of row vectors A with a batch of matrices B
    if A.shape[-1] != 1:
        # if A is a batch of matrices, trim to just take the first row
        A = A[:,:,A_idx:A_idx + 1]

    C = A.transpose(-1,-2) + B
    maxC = C.max(-1, keepdim=True).values
    return (C - maxC).exp().sum(-1,keepdim=True).log() + maxC

    
def stable_logexpmm(A, B, time_=False):
    assert B.shape[-2] == A.shape[-1]
    if time_:
        start = time.time()
    
    input_dim = B.shape[-1]
    output_dim = A.shape[-2]
    inner_dim = B.shape[-2]

    alphas = torch.arange(inner_dim, device=A.device)
    inputs = torch.arange(input_dim, device=A.device)
    outputs = torch.arange(output_dim, device=A.device)
    outputs, alphas, inputs = torch.meshgrid((outputs, alphas, inputs))
    logits = A[:,outputs,alphas] + B[:,alphas,inputs]
    max_logit = logits.max(-2, keepdim=True).values

    logits -= max_logit
    presum_values = logits.exp()
    values = presum_values.sum(-2)
    result = values.log() + max_logit.squeeze(-2)

    if time_:
        return result, time.time() - start
    else:
        return result
    

def unstable_inputs(args: argparse.Namespace):
    A = [[-232.8080, -232.8080, -339.6227, -232.8080, -233.7948, -232.8080,
          -232.8080, -233.4026],
         [-232.8129, -232.8129, -339.6276, -232.8129, -233.7997, -232.8129,
          -232.8129, -233.4075],
         [-232.8088, -232.8088, -339.6234, -232.8088, -233.7955, -232.8088,
          -232.8088, -233.4034],
         [-232.8114, -232.8114, -339.6261, -232.8114, -233.7982, -232.8114,
          -232.8114, -233.4060],
         [-232.8115, -232.8115, -339.6262, -232.8115, -233.7982, -232.8115,
          -232.8115, -233.4061],
         [-232.8137, -232.8137, -339.6284, -232.8137, -233.8005, -232.8137,
          -232.8137, -233.4083],
         [-232.8083, -232.8083, -339.6230, -232.8083, -233.7951, -232.8083,
          -232.8083, -233.4029],
         [-232.8084, -232.8084, -339.6231, -232.8084, -233.7952, -232.8084,
          -232.8084, -233.4030]]
    B = [[-105.8763,   -2.0794, -111.5282,   -2.0794, -109.5053, -102.4518,
          -120.6127, -121.5034],
         [-124.8149,   -2.0794, -105.6063,   -2.0794, -125.8682, -140.7194,
          -114.3827, -131.6997],
         [  -2.0794,   -2.0794,   -2.0794,   -2.0794,   -2.0794,   -2.0794,
            -2.0794,   -2.0794],
         [-140.2136,   -2.0794, -113.9071,   -2.0794, -127.7903, -117.2482,
          -122.3694, -129.3651],
         [ -75.4844,   -2.0794,   -2.0794,   -2.0794, -123.6696,   -2.0794,
           -2.0794,   -2.0794],
         [-133.8385,  -97.2946, -117.7260,   -2.0794, -149.4694, -138.0431,
          -123.8264, -138.2962],
         [-133.4516,  -81.5486, -134.3775,   -2.0794, -155.6442, -123.6420,
          -136.6097, -122.8662],
         [ -73.7120,   -2.0794,   -2.0794,   -2.0794, -108.0346,   -2.0794,
           -77.2179,  -89.1178]]
    return torch.tensor(A).to(args.device).unsqueeze(0), torch.tensor(B).to(args.device).unsqueeze(0)


def test_mm(args: argparse.Namespace):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.unstable:
        A, B = unstable_inputs(args)
    else:
        A = torch.empty(2,args.dim, args.dim, device=args.device).normal_() * args.multiplier
        B = torch.empty(2,args.dim, args.dim, device=args.device).uniform_() * args.multiplier

    C, time = logexpmm(A,B, time_=True)
    C_fast, time_ = fast_logexpmm(A,B, time_=True)
    C_stable, time_ = stable_logexpmm(A,B, time_=True)
    
    valid_fast = ((C - C_fast).abs() < 1e-5).all()
    valid_stable = ((C - C_stable).abs() < 1e-5).all()
    if args.print:
        print(C)
        print(C_fast)
        print(C_stable)
        print(time, time_)
        print(C - C_fast)
        print(C - C_stable)
    print(f'valid_fast: {valid_fast}, valid_stable: {valid_stable}')
    if not valid_fast or not valid_stable:
        import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()


def test_reduce(args: argparse.Namespace): 
    n = 40
    x = torch.ones(5).uniform_()
    x.requires_grad = True
    W = torch.empty(n*50*50,5).uniform_() / 10
    W.requires_grad = True

    A = (W @ x).view(n,50,50)
    b = reduce(fast_logexpmm, A.unsqueeze(1))
    c = dc_reduce(fast_logexpmm, A.unsqueeze(1))
    print(b)
    print(c)
    
    print(b-c)

    '''
    ad = torchviz.make_dot(a, params={'x': x, 'W': W, 'A': A})
    cd = torchviz.make_dot(c, params={'x': x, 'W': W, 'A': A})
    dd = torchviz.make_dot(d, params={'x': x, 'W': W, 'A': A})
    ad.render('linear.dot')
    cd.render('dc.dot')
    dd.render('dd.dot')
    '''


def test_bmv(args: argparse.Namespace):
    A = torch.empty(2, 1, 10).normal_() * 10
    B = torch.empty(2, 10, 10).normal_() * 10

    C = A.exp().bmm(B.exp()).log()
    C_ = fast_logexpmv(A, B)
    C__ = fast_logexpmv_right(A, B)

    print(C)
    print(C_)
    print(C__)
    print(C-C_)
    print(C-C__)

    import pdb; pdb.set_trace()
    

def test_order(args: argparse.Namespace):
    A = torch.randn(1,4,4)
    B = torch.randn(1,4,4)
    C = torch.randn(1,4,4)

    print(fast_logexpmm(A, fast_logexpmm(B, C)))
    print(fast_logexpmm(fast_logexpmm(A, B), C))
    
def test(args: argparse.Namespace):
    if args.test_reduce:
        test_reduce(args)
    elif args.test_bmv:
        test_bmv(args)
    elif args.test_order:
        test_order(args)
    else:
        test_mm(args)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--multiplier', type=float, default=1)
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--unstable', action='store_true')
    parser.add_argument('--test_reduce', action='store_true')
    parser.add_argument('--test_bmv', action='store_true')
    parser.add_argument('--test_order', action='store_true')
    args = parser.parse_args()
    test(args)
