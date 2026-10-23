"""Real-time guitar effect: audio I/O loop, offline render, and benchmark.

    python -m gtcn_live run --checkpoint best_BritPlexi.pt --embedding BritPlexi_embeddings.pt
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def load_effect(args):
    from .native import NativeEffect
    return NativeEffect.load(args.checkpoint, args.embedding, max_chunk=max(args.block, 4096))


def to_db(x):
    return 20 * np.log10(max(float(x), 1e-6))


class RealtimeRunner:
    """Per-block DSP chain: input gain -> neural effect -> wet/dry mix -> output gain -> clip."""

    def __init__(self, effect, in_gain_db=0.0, out_gain_db=0.0, mix=1.0):
        self.effect = effect
        self.in_gain = 10 ** (in_gain_db / 20)
        self.out_gain = 10 ** (out_gain_db / 20)
        self.mix = mix
        self.bypass = False
        self.in_peak = 0.0
        self.out_peak = 0.0
        self.busy = 0.0  # smoothed fraction of the block period spent computing
        self.xruns = 0
        self._out = np.empty(effect.max_chunk, dtype=np.float32)

    def process_block(self, dry):
        n = dry.shape[0]
        x = np.ascontiguousarray(dry * self.in_gain, dtype=np.float32)
        if self.bypass:
            wet = x
        else:
            wet = self.effect.process_np(x, self._out[:n])
            if self.mix < 1.0:
                wet = self.mix * wet + (1.0 - self.mix) * x
        y = np.clip(wet * self.out_gain, -1.0, 1.0)
        self.in_peak = max(self.in_peak * 0.9, float(np.abs(x).max()))
        self.out_peak = max(self.out_peak * 0.9, float(np.abs(y).max()))
        return y


def cmd_devices(_args):
    import sounddevice as sd
    print(sd.query_devices())


def cmd_run(args):
    import sounddevice as sd

    effect = load_effect(args)
    sr = effect.sample_rate
    runner = RealtimeRunner(effect, args.in_gain, args.out_gain, args.mix)

    in_dev = args.input_device if args.input_device is not None else sd.default.device[0]
    out_dev = args.output_device if args.output_device is not None else sd.default.device[1]
    in_info, out_info = sd.query_devices(in_dev, "input"), sd.query_devices(out_dev, "output")
    n_in = max(in_info["max_input_channels"], 1)
    n_out = max(min(out_info["max_output_channels"], 2), 1)
    if args.input_channel >= n_in:
        sys.exit(f"input device has only {n_in} channel(s); --input-channel is 0-based")

    effect.process_np(np.zeros(args.block, dtype=np.float32))  # warm up before audio starts
    effect.reset()

    block_ms = args.block / sr * 1000
    print(f"effect: {effect.name}   {sr} Hz   block {args.block} samples ({block_ms:.1f} ms)")
    print(f"input : [{in_dev}] {in_info['name']} (channel {args.input_channel})")
    print(f"output: [{out_dev}] {out_info['name']}")
    print("Use headphones to avoid feedback. Commands: wet <0-1> | in <dB> | out <dB> | bypass | q")

    def callback(indata, outdata, frames, time_info, status):
        t0 = time.perf_counter()
        if status:
            runner.xruns += 1
        y = runner.process_block(indata[:, args.input_channel])
        outdata[:] = y[:, None]
        runner.busy = 0.95 * runner.busy + 0.05 * (time.perf_counter() - t0) / (frames / sr)

    stop = threading.Event()

    def console():
        for line in sys.stdin:
            parts = line.split()
            if not parts:
                continue
            try:
                if parts[0] in ("q", "quit", "exit"):
                    break
                elif parts[0] == "wet":
                    runner.mix = min(max(float(parts[1]), 0.0), 1.0)
                elif parts[0] == "in":
                    runner.in_gain = 10 ** (float(parts[1]) / 20)
                elif parts[0] == "out":
                    runner.out_gain = 10 ** (float(parts[1]) / 20)
                elif parts[0] == "bypass":
                    runner.bypass = not runner.bypass
            except (ValueError, IndexError):
                print("\nbad command")
        stop.set()

    threading.Thread(target=console, daemon=True).start()

    with sd.Stream(samplerate=sr, blocksize=args.block, device=(in_dev, out_dev),
                   channels=(n_in, n_out), dtype="float32", latency=args.latency,
                   callback=callback) as stream:
        lat = stream.latency
        print(f"latency: in {lat[0] * 1000:.1f} ms + out {lat[1] * 1000:.1f} ms + block {block_ms:.1f} ms "
              f"~= {(lat[0] + lat[1]) * 1000 + block_ms:.1f} ms round trip. Ctrl-C to quit.")
        try:
            while not stop.is_set():
                time.sleep(0.5)
                print(f"\rin {to_db(runner.in_peak):6.1f} dBFS  out {to_db(runner.out_peak):6.1f} dBFS  "
                      f"cpu {runner.busy * 100:4.0f}%  xruns {runner.xruns}  "
                      f"{'BYPASS' if runner.bypass else effect.name}   ", end="", flush=True)
        except KeyboardInterrupt:
            pass
    print()


def cmd_render(args):
    effect = load_effect(args)
    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    if sr != effect.sample_rate:
        sys.exit(f"input is {sr} Hz but the model needs {effect.sample_rate} Hz; resample first")
    dry = np.ascontiguousarray(audio[:, 0] * 10 ** (args.in_gain / 20))
    out = np.concatenate([effect.process_np(np.ascontiguousarray(dry[i:i + args.block]))
                          for i in range(0, len(dry), args.block)])
    sf.write(args.output, np.clip(out * 10 ** (args.out_gain / 20), -1, 1), sr)
    print("wrote", args.output)


def cmd_bench(args):
    effect = load_effect(args)
    x = (np.random.randn(args.block) * 0.1).astype(np.float32)
    out = np.empty_like(x)
    for _ in range(20):
        effect.process_np(x, out)
    times = []
    for _ in range(500):
        t = time.perf_counter()
        effect.process_np(x, out)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    budget = args.block / effect.sample_rate * 1000
    print(f"block {args.block}: median {times[250]:.2f} ms, p99 {times[495]:.2f} ms, "
          f"max {times[-1]:.2f} ms, budget {budget:.2f} ms ({times[250] / budget * 100:.0f}% load)")


def cmd_embed(args):
    from .embedding import create_embedding

    try:
        name, stats = create_embedding(
            args.checkpoint, args.gtcn_source, args.dry, args.wet, args.output, args.name
        )
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"Created {name} embedding from {stats['num_inliers']}/{stats['num_references']} "
          f"reference windows: {args.output}")


def cmd_capture(args):
    import sounddevice as sd

    from .capture import ReferenceCapture
    from .embedding import load_model, save_embedding

    try:
        model, config, checkpoint_path = load_model(args.checkpoint, args.gtcn_source)
    except ValueError as exc:
        sys.exit(str(exc))
    sr = config["data"]["sample_rate"]
    need = config["data"].get("reference_seconds", 2.0)

    wet, wet_sr = sf.read(args.wet_file, dtype="float32", always_2d=True)
    if wet_sr != sr:
        sys.exit(f"{args.wet_file} is {wet_sr} Hz but the model needs {sr} Hz; resample first")
    wet = wet.mean(axis=1)[: int(args.seconds * sr)]
    if len(wet) < need * sr:
        sys.exit(f"need at least {need:g} s of wet audio, got {len(wet) / sr:.1f} s")

    in_dev = args.input_device if args.input_device is not None else sd.default.device[0]
    out_dev = args.output_device if args.output_device is not None else sd.default.device[1]
    in_info, out_info = sd.query_devices(in_dev, "input"), sd.query_devices(out_dev, "output")
    n_in = max(in_info["max_input_channels"], 1)
    n_out = max(min(out_info["max_output_channels"], 2), 1)
    if args.input_channel >= n_in:
        sys.exit(f"input device has only {n_in} channel(s); --input-channel is 0-based")

    cap = None
    in_gain = 10 ** (args.in_gain / 20)

    def callback(indata, outdata, frames, time_info, status):
        if status:
            cap.xruns += 1
        cap.process(indata[:, args.input_channel] * in_gain, outdata[:, 0])
        outdata[:, 1:] = outdata[:, :1]
        if cap.done:
            raise sd.CallbackStop

    stream = sd.Stream(samplerate=sr, blocksize=args.block, device=(in_dev, out_dev),
                       channels=(n_in, n_out), dtype="float32", latency=args.latency,
                       callback=callback)
    with stream:
        lat = stream.latency
        cap = ReferenceCapture(wet, round((lat[0] + lat[1]) * sr))
        print(f"input : [{in_dev}] {in_info['name']} (channel {args.input_channel})")
        print(f"output: [{out_dev}] {out_info['name']}   round-trip latency "
              f"{(lat[0] + lat[1]) * 1000:.1f} ms")
        print(f"Capturing {len(wet) / sr:.0f} s. Use headphones. Play your dry guitar along with the "
              "wet recording, as tightly as you can, and keep playing until told to stop.")
        stream.start()
        while stream.active:
            time.sleep(0.25)
            print(f"\r{cap.pos / sr:5.1f} / {len(wet) / sr:.0f} s   dry level "
                  f"{to_db(cap.peak):6.1f} dBFS   xruns {cap.xruns}   ", end="", flush=True)
    print()

    dry, wet = cap.aligned()
    if float(np.abs(dry).max()) < 1e-3:
        sys.exit("The recorded dry signal is silent. Check --input-device / --input-channel and the "
                 "input gain, then capture again.")
    if cap.xruns:
        print(f"warning: {cap.xruns} audio dropouts during capture; try --block 256 and capture again "
              "if the result sounds off.")
    name, stats = save_embedding(
        model, config, checkpoint_path, dry, wet, args.output, args.name,
        {"wet_reference": Path(args.wet_file).name, "source": "live capture"})
    print("\a\nDone - you can STOP playing now.")
    print(f"Saved {name} embedding ({stats['num_inliers']}/{stats['num_references']} windows) to {args.output}")
    print(f"Play with it: python -m gtcn_live run --checkpoint {args.checkpoint} --embedding {args.output} ...")


def build_parser():
    p = argparse.ArgumentParser(prog="gtcn_live", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--checkpoint", required=True, help="gTCN training checkpoint, e.g. best_BritPlexi.pt")
        sp.add_argument("--embedding", required=True, help="effect embedding, e.g. BritPlexi_embeddings.pt")
        sp.add_argument("--block", type=int, default=128, help="samples per audio block (latency/CPU trade-off)")
        sp.add_argument("--in-gain", type=float, default=0.0, help="input gain in dB")
        sp.add_argument("--out-gain", type=float, default=0.0, help="output gain in dB")

    sp = sub.add_parser("devices", help="list audio devices")
    sp.set_defaults(fn=cmd_devices)

    sp = sub.add_parser("run", help="live guitar in -> effect -> out")
    common(sp)
    sp.add_argument("--input-device", type=lambda s: int(s) if s.isdigit() else s)
    sp.add_argument("--output-device", type=lambda s: int(s) if s.isdigit() else s)
    sp.add_argument("--input-channel", type=int, default=0, help="0-based channel of the guitar input")
    sp.add_argument("--mix", type=float, default=1.0, help="wet amount 0..1")
    sp.add_argument("--latency", default="low", help="'low', 'high' or seconds")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("render", help="process a wav file with the streaming engine")
    common(sp)
    sp.add_argument("input")
    sp.add_argument("output")
    sp.set_defaults(fn=cmd_render)

    sp = sub.add_parser("bench", help="measure per-block compute time")
    common(sp)
    sp.set_defaults(fn=cmd_bench)

    sp = sub.add_parser("embed", help="create an effect embedding from paired dry/wet WAVs")
    sp.add_argument("--checkpoint", required=True, help="gTCN training checkpoint")
    sp.add_argument("--gtcn-source", required=True, help="path to gTCN/src containing model.py")
    sp.add_argument("--dry", required=True, help="dry reference WAV")
    sp.add_argument("--wet", required=True, help="matching wet reference WAV")
    sp.add_argument("--output", required=True, help="path for the generated embedding file")
    sp.add_argument("--name", help="display name for the effect (defaults to output filename)")
    sp.set_defaults(fn=cmd_embed)

    sp = sub.add_parser("capture", help="play a wet WAV while recording your dry playing, then create an embedding")
    sp.add_argument("--checkpoint", required=True, help="gTCN training checkpoint (use it again for 'run')")
    sp.add_argument("--gtcn-source", required=True, help="path to gTCN/src containing model.py")
    sp.add_argument("--wet-file", required=True, help="wet guitar recording to play along with")
    sp.add_argument("--output", required=True, help="path for the generated embedding file")
    sp.add_argument("--name", help="display name for the effect (defaults to output filename)")
    sp.add_argument("--seconds", type=float, default=30.0, help="how much of the wet file to capture")
    sp.add_argument("--input-device", type=lambda s: int(s) if s.isdigit() else s)
    sp.add_argument("--output-device", type=lambda s: int(s) if s.isdigit() else s)
    sp.add_argument("--input-channel", type=int, default=0, help="0-based channel of the guitar input")
    sp.add_argument("--in-gain", type=float, default=0.0, help="input gain in dB")
    sp.add_argument("--block", type=int, default=128)
    sp.add_argument("--latency", default="low", help="'low', 'high' or seconds")
    sp.set_defaults(fn=cmd_capture)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)
