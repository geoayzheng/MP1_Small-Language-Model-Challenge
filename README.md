# MP1_Small-Language-Model-Challenge
A project for HKU course DASE 7506
# MP1 code — installation and usage

Read [the project guide](../guide/GUIDE.md) for the assignment, assessment, deadlines and peer review. This README contains the running instructions and technical rules. The package has only these two documents.

All commands below run from **code/**. Data and the tokenizer are included. No API key, pretrained weights or additional dataset download is needed; after installing dependencies, training and evaluation work offline.

## 1. Install

Use **Python 3.12**. From the extracted package directory:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/base_smoke/checkpoint.pt --device cpu --precision fp32 --split validation   # T_base = "seconds": 4.629072125000675
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

```bash
python train_student.py --config configs/s320.json --steps 100 --run-dir runs/s320_smoke --device cuda --precision bf16
python evaluate.py --checkpoint runs/s320_smoke/checkpoint_final.pt --device cpu --precision fp32 --split validation  #T = "seconds": 19.353627052001684 ≤ 4.6×T_base（cache 已含在内）
``` 

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C, measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**My model** — `student.py` and supporting files, then:

```bash
# 训练teachers （两两并行）
python train_student.py --config configs/t320.json --steps 48000 --batch-size 32 --seed 17 \
    --eval-every 2400 --run-dir runs/t17 --device cuda --precision bf16
python train_student.py --config configs/t320.json --steps 48000 --batch-size 32 --seed 29 \
    --eval-every 2400 --run-dir runs/t29 --device cuda --precision bf16
```

```bash
# teacher 缓存
python make_teacher_cache.py --checkpoints runs/t17/checkpoint_ema.pt runs/t29/checkpoint_ema.pt runs/s5_main/checkpoint.pt --out runs/kd_cache.npz --device cuda 
    #top-1 agreement  >0.5
```

```bash
# 72k 主候选
python train_student.py --config configs/s320.json --steps 72000 --batch-size 32 --seed 17 --eval-every 2400 --kd-cache runs/kd_cache.npz --kd-alpha 0.5  --run-dir runs/s_both48 --device cuda --precision bf16
python train_student.py --config configs/s320.json --steps 72000 --batch-size 32 --seed 17 --eval-every 2400 --run-dir runs/s_dir48 --device cuda --precision bf16
```

```bash
# 19.2k 机制矩阵（两两并行）
python train.py --implementation model --device cuda --precision bf16 --steps 19200 --batch-size 32 --seed 17 --run-dir runs/s0_19k          # 等 token 基线

python train_student.py --config configs/t320.json --steps 19200 --batch-size 32 --seed 17 --eval-every 800 --run-dir runs/arm_no19k --device cuda --precision bf16

python train_student.py --config configs/s320.json --steps 19200 --batch-size 32 --seed 17 --eval-every 800 --run-dir runs/arm_cache19k --device cuda --precision bf16

python train_student.py --config configs/t320.json --steps 19200 --batch-size 32 --seed 17 --eval-every 800 --kd-cache runs/kd_cache.npz --kd-alpha 0.5 --run-dir runs/arm_kd19k --device cuda --precision bf16

python train_student.py --config configs/s320.json --steps 19200 --batch-size 32 --seed 17 --eval-every 800 --kd-cache runs/kd_cache.npz --kd-alpha 0.5 --run-dir runs/arm_both19k --device cuda --precision bf16

python train.py --implementation model --device cuda --precision bf16 --steps 48000 --batch-size 32 --seed 17 --run-dir runs/s0_48k          # 最终候选的等 token 基线
```

```bash
# 4800-step 快速架构消融
python train_student.py --config configs/s320.json          --steps 4800 --batch-size 32 --seed 17 --eval-every 1600 --run-dir runs/ab_main4800   --device cuda --precision bf16

python train_student.py --config configs/ab_gelu4800.json   --steps 4800 --batch-size 32 --seed 17 --eval-every 1600 --run-dir runs/ab_gelu4800   --device cuda --precision bf16

python train_student.py --config configs/ab_drop4800.json   --steps 4800 --batch-size 32 --seed 17 --eval-every 1600 --run-dir runs/ab_drop4800   --device cuda --precision bf16

python train_student.py --config configs/ab_untied4800.json --steps 4800 --batch-size 32 --seed 17 --eval-every 1600 --run-dir runs/ab_untied4800 --device cuda --precision bf16

python train_student.py --config configs/t320.json --steps 4800 --batch-size 32 --seed 17 --eval-every 1600 --run-dir runs/ab_nocache4800 --device cuda --precision bf16
```

```bash
# 选优（全部 validation）
for r in s_both48 s_dir48 arm_both19k; do for v in final ema avg; do
  python evaluate.py --checkpoint runs/$r/checkpoint_$v.pt --device cpu --precision fp32 --split validation
done; done
```

```bash
# 冻结 + 唯一一次 test
python evaluate.py --checkpoint runs/s_both48/checkpoint_avg.pt --device cpu --precision fp32 --split test   # 换成选优结果

```

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` (or the corresponding device/split name) and per-window losses. Submit the **bpb** value from the complete-test JSON, not token perplexity or validation BPB. Default evaluation is FP32. Add `--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver overhead is additional.

## 3. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons. |
| `student.py`, `train.py`,`train_student.py` | My model factory and training recipe; add supporting code as needed. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; keep unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; keep unchanged. |
| `tests/test_contract.py` | Checks your model's causality, normalization, independence and gradients. |
| `RUN_LOG_TEMPLATE.csv` | Optional experiment-log template. |
| `PACKAGE_MANIFEST.json` | Release hashes; paths are relative to the package root containing code/ and guide/. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

Measure all three limits for the same frozen predictor:

- **CPU time ≤5× baseline:**
- **Peak RAM ≤4 GiB:**
- **Inference assets ≤64 MiB uncompressed:** 


## 5. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.