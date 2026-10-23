"""ctypes wrapper around the C streaming kernel (gtcn_rt.c), auto-built on first use."""

import ctypes
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent / "csrc"
SRC = HERE / "gtcn_rt.c"
LIB = HERE / ("libgtcn_rt.dylib" if sys.platform == "darwin" else "libgtcn_rt.so")


def build_native(force=False):
    if LIB.exists() and not force and LIB.stat().st_mtime >= SRC.stat().st_mtime:
        return LIB
    if sys.platform == "darwin":
        cmd = ["clang", "-O3", "-ffast-math", "-DACCELERATE_NEW_LAPACK", "-shared", "-fPIC", str(SRC),
               "-framework", "Accelerate", "-o", str(LIB)]
        if platform.machine() == "arm64":
            cmd.insert(1, "-mcpu=native")
    elif sys.platform.startswith("linux"):
        cmd = ["cc", "-O3", "-ffast-math", "-march=native", "-shared", "-fPIC", str(SRC),
               "-lopenblas", "-lm", "-o", str(LIB)]
    else:
        raise RuntimeError("native kernel is only supported on macOS and Linux")
    subprocess.run(cmd, check=True)
    return LIB


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class NativeEffect:
    """Same interface as engine.StreamingEffect, but runs in C (no Python in the hot loop)."""

    def __init__(self, backbone_state, model_cfg, z, max_chunk=1024):
        lib = ctypes.CDLL(str(build_native()))
        lib.gtcn_create.restype = ctypes.c_void_p
        lib.gtcn_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.gtcn_reset.argtypes = [ctypes.c_void_p]
        lib.gtcn_destroy.argtypes = [ctypes.c_void_p]
        self.lib = lib
        self.max_chunk = max_chunk
        self.sample_rate = None
        self.name = "effect"

        rc, sc = model_cfg["residual_channels"], model_cfg["skip_channels"]
        k, lpb = model_cfg["kernel_size"], model_cfg["layers_per_block"]
        L = model_cfg["num_blocks"] * lpb
        z = z.float().view(1, -1)
        s = {key: v.float() for key, v in backbone_state.items()}
        keep = []  # numpy arrays must outlive the C engine

        def arr(t):
            a = np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)
            keep.append(a)
            return a

        def taps_major(w):  # (out, in, k) -> (out, k*in)
            return arr(torch.cat([w[:, :, j] for j in range(w.shape[-1])], dim=1))

        pe1w, pe2w = s["performance_encoder.conv1.weight"], s["performance_encoder.conv2.weight"]
        pe_k = pe1w.shape[-1]
        a_pe1w, a_pe1b = taps_major(pe1w), arr(s["performance_encoder.conv1.bias"])
        a_pe2w, a_pe2b = taps_major(pe2w), arr(s["performance_encoder.conv2.bias"])

        gW, gb, Wr, br, dil = [], [], [], [], []
        for i in range(L):
            pre = f"renderer.blocks.{i}."
            w, b = s[pre + "conv.weight"], s[pre + "conv.bias"]
            gamma, beta = torch.nn.functional.linear(z, s[pre + "film.weight"], s[pre + "film.bias"]).view(-1).chunk(2)
            w = w * (1 + gamma).view(-1, 1, 1)
            b = b * (1 + gamma) + beta
            # sigmoid(x) = 0.5 + 0.5*tanh(x/2): pre-halve the gate rows
            w = torch.cat([w[:rc], w[rc:] * 0.5])
            b = torch.cat([b[:rc], b[rc:] * 0.5])
            gW.append(_ptr(taps_major(w)))
            gb.append(_ptr(arr(b)))
            Wr.append(_ptr(arr(torch.cat([s[pre + "res_conv.weight"], s[pre + "skip_conv.weight"]], 0)[:, :, 0])))
            br.append(_ptr(arr(torch.cat([s[pre + "res_conv.bias"], s[pre + "skip_conv.bias"]], 0))))
            dil.append(2 ** (i % lpb))

        o1w, o1b = arr(s["renderer.output_conv1.weight"][:, :, 0]), arr(s["renderer.output_conv1.bias"])
        o2w, o2b = arr(s["renderer.output_conv2.weight"][:, :, 0]), arr(s["renderer.output_conv2.bias"])

        vp = ctypes.c_void_p
        self._arrays = (vp * L)(*gW), (vp * L)(*gb), (vp * L)(*Wr), (vp * L)(*br)
        self._dil = (ctypes.c_int * L)(*dil)
        self._keep = keep
        self._h = lib.gtcn_create(
            rc, sc, L, max_chunk, pe_k, _ptr(a_pe1w), _ptr(a_pe1b), _ptr(a_pe2w), _ptr(a_pe2b),
            k, self._dil, *self._arrays, _ptr(o1w), _ptr(o1b), _ptr(o2w), _ptr(o2b))
        self._h = ctypes.c_void_p(self._h)
        self.receptive_field = 1 + model_cfg["num_blocks"] * (k - 1) * (2 ** lpb - 1) + 2 * (pe_k - 1)

    @classmethod
    def load(cls, checkpoint_path, embedding_path, max_chunk=1024):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        emb = torch.load(embedding_path, map_location="cpu", weights_only=False)
        want = emb.get("checkpoint")
        if want and Path(want).name != Path(checkpoint_path).name:
            print(f"warning: embedding was computed with {Path(want).name}, but you loaded "
                  f"{Path(checkpoint_path).name}; the sound may be wrong.")
        eng = cls(ckpt["model_state_dict"], ckpt["config"]["model"], emb["z"], max_chunk)
        eng.sample_rate = ckpt["config"]["data"]["sample_rate"]
        eng.name = emb.get("timbre", Path(embedding_path).stem)
        return eng

    def reset(self):
        self.lib.gtcn_reset(self._h)

    def process_np(self, x, out=None):
        """x: contiguous float32 ndarray of N samples -> float32 ndarray of N wet samples."""
        n = x.shape[0]
        if n > self.max_chunk:
            raise ValueError(f"chunk {n} > max_chunk {self.max_chunk}")
        if out is None:
            out = np.empty(n, dtype=np.float32)
        self.lib.gtcn_process(self._h, _ptr(x), _ptr(out), n)
        return out

    def process(self, x):
        a = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        return torch.from_numpy(self.process_np(a))

    def __del__(self):
        if getattr(self, "_h", None) and getattr(self, "lib", None):
            self.lib.gtcn_destroy(self._h)
            self._h = None
