import numpy as np
import pytest
import torch

from gtcn_live.engine import StreamingEffect
from gtcn_live.native import NativeEffect

CFG = dict(residual_channels=8, skip_channels=8, kernel_size=3, num_blocks=2,
           layers_per_block=5, cond_dim=6, encoder_base_channels=4)


def random_state(seed=0):
    g = torch.Generator().manual_seed(seed)
    rc, sc, cd, pk = 8, 8, 6, 15

    def r(*shape, s=0.3):
        return torch.randn(*shape, generator=g) * s

    st = {
        "performance_encoder.conv1.weight": r(rc, 1, pk), "performance_encoder.conv1.bias": r(rc),
        "performance_encoder.conv2.weight": r(rc, rc, pk, s=0.1), "performance_encoder.conv2.bias": r(rc),
        "renderer.output_conv1.weight": r(sc, sc, 1), "renderer.output_conv1.bias": r(sc),
        "renderer.output_conv2.weight": r(1, sc, 1), "renderer.output_conv2.bias": r(1),
    }
    for i in range(CFG["num_blocks"] * CFG["layers_per_block"]):
        p = f"renderer.blocks.{i}."
        st[p + "conv.weight"], st[p + "conv.bias"] = r(2 * rc, rc, 3), r(2 * rc)
        st[p + "film.weight"], st[p + "film.bias"] = r(4 * rc, cd), r(4 * rc)
        st[p + "res_conv.weight"], st[p + "res_conv.bias"] = r(rc, rc, 1), r(rc)
        st[p + "skip_conv.weight"], st[p + "skip_conv.bias"] = r(sc, rc, 1), r(sc)
    return st, r(cd, s=1.0)


def run_chunks(eng, x, chunk, native):
    outs = []
    for i in range(0, len(x), chunk):
        c = x[i:i + chunk]
        outs.append(torch.from_numpy(eng.process_np(c.numpy().copy())) if native else eng.process(c))
    return torch.cat(outs)


@pytest.mark.parametrize("chunk", [1, 7, 64, 128, 300])
def test_native_matches_torch_engine_any_chunking(chunk):
    st, z = random_state()
    x = torch.randn(1500) * 0.3
    ref = StreamingEffect(st, CFG, z)
    want = run_chunks(ref, x, 1500, native=False)
    nat = NativeEffect(st, CFG, z, max_chunk=512)
    got = run_chunks(nat, x, chunk, native=True)
    assert torch.allclose(got, want, atol=1e-5)


def test_reset_restores_initial_state():
    st, z = random_state()
    x = torch.randn(400) * 0.3
    nat = NativeEffect(st, CFG, z, max_chunk=512)
    first = run_chunks(nat, x, 100, native=True)
    nat.reset()
    again = run_chunks(nat, x, 100, native=True)
    assert np.array_equal(first.numpy(), again.numpy())
