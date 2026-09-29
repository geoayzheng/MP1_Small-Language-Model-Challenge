"""Top-k teacher log-prob cache over ALL training positions, from one or more
self-trained checkpoints (log-probs averaged = geometric-mean ensemble).
Training-time artifact only; NOT part of submitted inference assets."""
import argparse
from pathlib import Path
import numpy as np
import torch
from common import load_data, make_model, setup, autocast, windows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--topk', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    device, precision = setup(args.device, 'bf16' if args.device == 'cuda' else 'fp32', 4)
    data = load_data()
    n = len(data['train'][0])
    W = (n - 1 + 255) // 256
    K = args.topk
    ids_acc = np.zeros((W, 256, K), dtype=np.int32)
    lp_acc = np.zeros((W, 256, K), dtype=np.float16)

    models = []
    for ck in args.checkpoints:
        c = torch.load(ck, map_location='cpu', weights_only=True)
        m, _ = make_model(c['implementation'], c['config'], device)
        m.load_state_dict(c['model'])
        m.eval()
        models.append(m)
        print(f'loaded teacher {ck} ({sum(q.numel() for q in m.parameters())} params)')

    gi = 0
    with torch.no_grad():
        for xb, _yb in windows(data['train'][0], batch_size=args.batch_size):
            xb = xb.to(device)
            avg = None
            for m in models:
                with autocast(device, precision):
                    lp = m.predict_log_probs(xb)
                avg = lp if avg is None else avg + lp
            avg = avg / len(models)
            v, idx = avg.topk(K, dim=-1)
            v = v - v.logsumexp(-1, keepdim=True)
            B = xb.shape[0]
            ids_acc[gi:gi + B] = idx.cpu().numpy().astype(np.int32)
            lp_acc[gi:gi + B] = v.cpu().numpy().astype(np.float16)
            gi += B
            if gi % 2048 < args.batch_size:
                print(f'{gi}/{W} windows', flush=True)
    assert gi == W, (gi, W)
    np.savez(args.out, ids=ids_acc, logps=lp_acc)
    t = data['train'][0]
    hit = sum(ids_acc[p // 256, p % 256, 0] == t[p + 1].item()
              for p in torch.randint(0, n - 1, (1000,)).tolist()) / 1000
    print(f'saved {args.out}; top-1 teacher agreement: {hit:.3f}')


if __name__ == '__main__':
    main()