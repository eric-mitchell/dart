import argparse
import torch
import time


def logexpmm(A, B, time_=False):
    start = time.time()
    result = (A.exp() @ B.exp()).log()

    return result, time.time() - start


def fast_logexpmm(A, B, time_=False):
    if time_:
        start = time.time()

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
    

def test(args: argparse.Namespace):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    A = torch.empty(2,args.dim, args.dim, device=args.device).uniform_() * args.multiplier
    B = torch.empty(2,args.dim, args.dim, device=args.device).uniform_() * args.multiplier

    C, time = logexpmm(A,B, time_=True)
    C_stable, time_ = stable_logexpmm(A,B, time_=True)

    valid = ((C - C_stable).abs() < 1e-5).all()
    if args.print:
        #print(C)
        #print(C_stable)
        print(time, time_)
        print(C - C_stable)
    print(valid)
    if not valid:
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--multiplier', type=float, default=1)
    parser.add_argument('--print', action='store_true')
    args = parser.parse_args()
    test(args)
