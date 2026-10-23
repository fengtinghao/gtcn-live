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
python3.13 -m venv gtcnlive
source gtcnlive/bin/activate
python -m pip install -r requirements.txt     # needs clang (Xcode CLT) on macOS; the kernel auto-builds on first run
```

Linux needs `cc` + OpenBLAS (`libopenblas-dev`); Windows is not supported.

## Folders

- `checkpoints/` — gTCN checkpoints (`last.pt`, `best_*.pt`); not tracked by Git (36 MB each)
- `embeddings/` — effect embeddings (`<Effect>_embeddings.pt`); small, tracked by Git

An embedding only works with the checkpoint it was created with. The five shipped embeddings
(`BritPlexi`, `FenderDeluxe`, `FenderDouble`, `Alpaca_Rogue`, `minosaur`) pair with the
matching `checkpoints/best_<Effect>.pt`. Embeddings you create yourself should use `checkpoints/last.pt`.

## Workflow

**1. Find your audio interface**

```bash
python -m gtcn_live devices          # note the device ID, e.g. 6 for "Balance"
```

**2. Create an effect embedding by playing along with a wet recording**

The app plays a wet guitar recording (the sound of the effect) while it records your dry guitar
from the interface. When enough audio has been captured it creates and saves the embedding, then
tells you to stop playing.

```bash
python -m gtcn_live capture \
    --checkpoint checkpoints/last.pt \
    --gtcn-source ../gTCN/src \
    --wet-file references/wet.wav \
    --output embeddings/MyEffect_embeddings.pt --name MyEffect \
    --input-device 6 --output-device 6 --input-channel 1
```

Tips for a good embedding:
- Use **headphones**, and play the **same part** as the wet recording (same notes and rhythm).
  Being slightly off in time barely matters; playing different notes degrades the result.
- Use a wet WAV at the checkpoint's sample rate (44.1 kHz) with at least 2 s of audio. It captures the
  first `--seconds` (default 30), and longer, varied playing gives a more robust embedding.
- Keep the dry level healthy (peaks around -12 to -6 dBFS). Watch the `dry level` readout while playing;
  adjust with `--in-gain` or the interface gain. A silent input is rejected.
- Dropouts (`xruns`) hurt alignment; if you see them, capture again with `--block 256`.

If you already have a matching dry/wet pair of WAVs (same performance, sample-aligned, same length):

```bash
python -m gtcn_live embed --checkpoint checkpoints/last.pt --gtcn-source ../gTCN/src \
    --dry references/dry.wav --wet references/wet.wav --output embeddings/MyEffect_embeddings.pt --name MyEffect
```

**3. Play guitar through the effect in real time**

```bash
python -m gtcn_live run \
    --checkpoint checkpoints/last.pt \
    --embedding embeddings/MyEffect_embeddings.pt \
    --input-device 6 --output-device 6 --input-channel 1 --block 128
```

Or one of the shipped effects:

```bash
python -m gtcn_live run --checkpoint checkpoints/best_BritPlexi.pt \
    --embedding embeddings/BritPlexi_embeddings.pt --input-device 6 --output-device 6 --input-channel 1
```

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
- `gtcn_live/capture.py`, `gtcn_live/embedding.py` — play-along capture and embedding creation
