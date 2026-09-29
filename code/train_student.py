"""Student recipe: config-driven LR + EMA + late-checkpoint averaging + optional KD.
KD reads a top-k teacher cache (make_teacher_cache.py); teachers are self-trained.
Saves checkpoint_final.pt / checkpoint_ema.pt / checkpoint_avg.pt and validates all.
"""
import argparse, copy, json, math, time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def lr_at(step, total, base, warmup, min_frac):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * (min_frac + (1 - min_frac) * .5 * (1 + math.cos(math.pi * prog)))


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/w320.json')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=48000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0)
    p.add_argument('--kd-cache', type=Path, default=None)
    p.add_argument('--kd-alpha', type=float, default=0.5)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')

    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    base_lr = float(config.get('lr', .001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(config.get('weight_decay', .1)))
    warmup = int(config.get('warmup_steps', 100))
    min_frac = float(config.get('min_lr_frac', 0.1))
    ema_decay = float(config.get('ema_decay', 0.0))
    snap_interval = int(config.get('snap_interval', 0))
    snap_count = int(config.get('snap_count', 4))
    ema = ({k: v.detach().clone().float() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point} if ema_decay > 0 else None)
    snapshots = deque(maxlen=snap_count) if snap_interval > 0 else None

    kd = None
    if args.kd_cache:
        z = np.load(args.kd_cache, mmap_mode='r')
        kd = (z['ids'], z['logps'])

    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared
    started = time.perf_counter()
    history, validation_history = [], []
    intermediate_validation_seconds = 0.

    for step in range(args.steps):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step, args.steps, base_lr, warmup, min_frac)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = model(batch[:, :-1])
        logp = F.log_softmax(logits.float(), dim=-1)
        loss = F.cross_entropy(logp.flatten(0, 1), batch[:, 1:].flatten())
        if kd is not None:
            ids_np, lp_np = kd
            pos = starts[:, None] + torch.arange(256, device=device)
            w = (pos // 256).cpu().numpy()
            j = (pos % 256).cpu().numpy()
            t_ids = torch.from_numpy(np.ascontiguousarray(ids_np[w, j])).to(device).long()
            t_lp = torch.from_numpy(np.ascontiguousarray(lp_np[w, j])).to(device).float()
            s_top = logp.gather(-1, t_ids)
            loss = (1 - args.kd_alpha) * loss \
                 - args.kd_alpha * (t_lp.exp() * s_top).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                msd = model.state_dict()
                for k, v in ema.items():
                    v.mul_(ema_decay).add_(msd[k].detach().float(), alpha=1 - ema_decay)
        if snapshots is not None and (step + 1) % snap_interval == 0:
            with torch.no_grad():
                snapshots.append({k: v.detach().float().cpu().clone()
                                  for k, v in model.state_dict().items()})
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {'step': step + 1, 'loss': loss.item(),
                   'seconds': time.perf_counter() - started - intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            intermediate = score(model, *data['validation'], device, 'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step': step + 1, **intermediate})
            print(json.dumps({'validation': validation_history[-1]}), flush=True)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - intermediate_validation_seconds

    def save_ckpt(state_dict, path):
        torch.save({'protocol': PROTOCOL, 'implementation': args.implementation,
                    'config': config,
                    'model': {k: v.detach().cpu() for k, v in state_dict.items()},
                    'seed': args.seed,
                    'train_tokens': args.steps * args.batch_size * 256}, path)

    candidates = {'final': model.state_dict()}
    if ema is not None:
        sd = model.state_dict()
        candidates['ema'] = {k: (ema[k].to(sd[k].dtype) if k in ema else sd[k]) for k in sd}
    if snapshots is not None and len(snapshots) >= 2:
        candidates['avg'] = {k: sum(s[k] for s in snapshots) / len(snapshots)
                             for k in snapshots[0]}

    validations = {}
    for name, sd in candidates.items():
        m = copy.deepcopy(model)
        m.load_state_dict(sd)
        save_ckpt(sd, args.run_dir / f'checkpoint_{name}.pt')
        m.to(device)
        v = score(m, *data['validation'], device, 'fp32')
        v.pop('window_nll_nats')
        validations[name] = v
        print(json.dumps({f'validation_{name}': v}), flush=True)

    result = {'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
              'seed': args.seed, 'steps': args.steps, 'batch_size': args.batch_size,
              'parameters': sum(q.numel() for q in model.parameters()),
              'precision': precision, 'train_tokens': args.steps * args.batch_size * 256,
              'preparation_seconds': preparation_seconds, 'train_seconds': train_seconds,
              'history': history, 'validation_history': validation_history,
              'intermediate_validation_seconds': intermediate_validation_seconds,
              'process_seconds': time.perf_counter() - total_started,
              'torch_version': str(torch.__version__), 'threads': args.threads,
              'implementation_sha256': implementation_sha,
              'kd_cache': str(args.kd_cache) if args.kd_cache else None,
              'kd_alpha': args.kd_alpha if args.kd_cache else None,
              'validations': validations, **device_metrics(device)}
    (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'process_seconds': result['process_seconds']}), flush=True)


if __name__ == '__main__':
    main()