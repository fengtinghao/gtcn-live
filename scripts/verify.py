"""Check streaming output against the original offline gTCN model, and measure speed.

Usage: python scripts/verify.py --gtcn ../gTCN --checkpoint best_BritPlexi.pt --embedding BritPlexi_embeddings.pt
"""

import argparse
import sys
import time
from pathlib import Path

import torch
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtcn_live.engine import StreamingEffect  # noqa: E402
from gtcn_live.native import NativeEffect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtcn", required=True, help="path to the gTCN repo (for the reference model)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--engine", choices=["native", "torch"], default="native")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.gtcn) / "src"))
    from model import GTCNModel  # noqa: E402

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ref = GTCNModel.from_config(ckpt["config"]["model"])
    ref.load_state_dict(ckpt["model_state_dict"])
    ref.eval()
    emb = torch.load(args.embedding, map_location="cpu", weights_only=False)

    torch.manual_seed(0)
    x = torch.randn(20000) * 0.2
    with torch.no_grad():
        want = ref.render_with_z(x.view(1, 1, -1), emb["z"].view(1, -1).float()).view(-1)

    eng = (NativeEffect if args.engine == "native" else StreamingEffect).load(args.checkpoint, args.embedding)
    got = torch.cat([eng.process(x[i:i + args.chunk]) for i in range(0, len(x), args.chunk)])
    print("max abs diff vs offline:", (got - want).abs().max().item())

    eng.reset()
    n = 200
    chunk = torch.randn(args.chunk) * 0.2
    for _ in range(10):
        eng.process(chunk)
    t = time.perf_counter()
    for _ in range(n):
        eng.process(chunk)
    ms = (time.perf_counter() - t) / n * 1000
    budget = args.chunk / eng.sample_rate * 1000
    print(f"chunk={args.chunk}: {ms:.2f} ms per chunk, budget {budget:.2f} ms")


if __name__ == "__main__":
    main()
