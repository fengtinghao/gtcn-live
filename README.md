# gtcn-live

Play a pre-trained [gTCN](https://github.com/fengtinghao/gTCN) guitar effect in real time:
guitar in → neural amp/effect → wet sound out, at ~3 ms processing block size.

```
guitar (audio interface) --> gtcn_live (C/BLAS streaming TCN) --> headphones / monitors
```

## How it is fast enough

The gTCN renderer is a 40-layer causal dilated TCN (186 ms receptive field, ~1.4 M MACs per
sample, ≈124 GFLOP/s at 44.1 kHz). A stock PyTorch conv1d runs at ~5× slower than real time on an M1 Ultra.
This app instead:

- keeps per-layer history, so each audio block only computes its own new samples
  (output is identical to offline rendering: max diff ~1e-7);
- folds the fixed effect embedding `z` (FiLM) into the conv weights at load time, and fuses the
  res/skip 1×1 convs;
- runs every causal conv as a single GEMM (Accelerate/AMX on macOS) from a small C kernel
  (`gtcn_live/csrc/gtcn_rt.c`) with no allocation or Python in the audio loop.

Measured on an Apple M1 Ultra, single thread: **1.5 ms per 128-sample block (2.9 ms budget)**,
0.9 ms per 64-sample block.

## Install

```bash
pip install -r requirements.txt     # needs clang (Xcode CLT) on macOS; the kernel auto-builds on first run
```

Linux needs `cc` + OpenBLAS (`libopenblas-dev`); Windows is not supported.

## Use

Effect embeddings are included in [`embeddings/`](./embeddings/). The full gTCN training
checkpoints are not included; download the matching checkpoint from the
[gTCN repo](https://github.com/fengtinghao/gTCN) before running an effect:

```bash
python -m gtcn_live devices
python -m gtcn_live run \
    --checkpoint ../gTCN/checkpoints/best_BritPlexi.pt \
    --embedding embeddings/BritPlexi_embeddings.pt \
    --input-device "Scarlett 2i2" --input-channel 0 --block 128
```

Included embeddings: `BritPlexi`, `FenderDeluxe`, `FenderDouble`, `Alpaca_Rogue`,
and `minosaur`. Use each with its corresponding gTCN checkpoint.

**Use headphones** (or you will get feedback). Type commands in the terminal while playing:
`wet 0.8`, `in 6` (input gain dB), `out -3`, `bypass`, `q`.

The status line shows input/output peak, CPU load (% of the block period) and xruns. Use
`--block 64` for lower latency, `--block 256` if you hear dropouts.

Latency = audio-driver input + output latency + one block. The processing contributes only ~3 ms; the
rest is your audio hardware (a dedicated interface gives a few ms, built-in/Bluetooth devices can be 30–100+ ms).
Use an interface with a **44.1 kHz** clock if possible — models were trained at 44.1 kHz.
Input level matters: trim with `--in-gain` so the guitar peaks look like the training data (the
effect is a nonlinear amp model, so level changes the tone).

Other commands:

```bash
python -m gtcn_live render --checkpoint ... --embedding ... in.wav out.wav   # offline, same engine
python -m gtcn_live bench  --checkpoint ... --embedding ... --block 128      # per-block timing
python scripts/verify.py --gtcn ../gTCN --checkpoint ... --embedding ...    # compare against gTCN's offline model
python -m pytest tests
```

## Layout

- `gtcn_live/native.py`, `csrc/gtcn_rt.c` — real-time engine (used by the app)
- `gtcn_live/engine.py` — pure-PyTorch streaming engine (reference / fallback, too slow for live use)
- `gtcn_live/app.py` — audio loop, CLI
