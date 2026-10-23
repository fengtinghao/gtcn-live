"""Stateful streaming inference engine for a trained gTCN effect.

The offline gTCN model renders a whole clip with zero left-padding in every
causal conv. Here every conv keeps its own input history between calls, so a
stream processed in arbitrary chunks gives the *same* samples as offline
rendering (up to float error) without re-computing the receptive field for
every block.

Speed-ups that do not change the output:
  * the effect embedding z is fixed, so FiLM (gamma, beta) is folded into the
    dilated conv weights/bias once at load time;
  * res_conv and skip_conv (both 1x1 on the same input) are fused into one conv.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

SQRT_HALF = 0.7071067811865476


class _CausalConv:
    """Causal (dilated) conv over [history | chunk], expressed as one matmul.

    Conv1d on CPU is slow in PyTorch; stacking the K dilated taps into a
    (K*C_in, N) matrix and doing a single GEMM hits Apple's AMX/Accelerate fast
    path and is >10x faster at these tiny sizes.
    """

    def __init__(self, weight, bias, dilation):
        out_ch, in_ch, k = weight.shape
        self.k, self.dilation = k, dilation
        self.weight = torch.cat([weight[:, :, j] for j in range(k)], dim=1).contiguous()
        self.bias = bias.view(-1, 1).contiguous()
        self.pad = (k - 1) * dilation
        self.state = torch.zeros(in_ch, self.pad, dtype=weight.dtype, device=weight.device)

    def reset(self):
        self.state = torch.zeros_like(self.state)

    def __call__(self, x):
        n, d = x.shape[-1], self.dilation
        full = torch.cat([self.state, x], dim=-1)  # (C, pad + N)
        self.state = full[:, n:].clone()
        taps = torch.cat([full[:, j * d: j * d + n] for j in range(self.k)], dim=0)
        return torch.addmm(self.bias, self.weight, taps)


class StreamingEffect:
    """A loaded effect (backbone weights + fixed embedding z), ready to stream."""

    def __init__(self, backbone_state, model_cfg, z, device="cpu", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.sample_rate = None
        self.name = "effect"

        rc = model_cfg["residual_channels"]
        k = model_cfg["kernel_size"]
        lpb = model_cfg["layers_per_block"]
        n_layers = model_cfg["num_blocks"] * lpb

        def p(name):
            return backbone_state[name].to(self.device, dtype)

        z = z.to(self.device, dtype).view(1, -1)

        self.pe1 = _CausalConv(p("performance_encoder.conv1.weight"),
                               p("performance_encoder.conv1.bias"), 1)
        self.pe2 = _CausalConv(p("performance_encoder.conv2.weight"),
                               p("performance_encoder.conv2.bias"), 1)

        self.blocks = []
        for i in range(n_layers):
            pre = f"renderer.blocks.{i}."
            dilation = 2 ** (i % lpb)
            w, b = p(pre + "conv.weight"), p(pre + "conv.bias")
            film = F.linear(z, p(pre + "film.weight"), p(pre + "film.bias")).view(-1)
            gamma, beta = film.chunk(2)  # each 2*rc, applied to the conv output channels
            w = w * (1 + gamma).view(-1, 1, 1)
            b = b * (1 + gamma) + beta
            # tanh(f) = 2*sigmoid(2f) - 1: scale the filter rows by 2 so one sigmoid
            # call covers both halves (PyTorch's CPU tanh is ~3x slower than sigmoid).
            w = torch.cat([w[:rc] * 2, w[rc:]]); b = torch.cat([b[:rc] * 2, b[rc:]])
            conv = _CausalConv(w.contiguous(), b.contiguous(), dilation)
            rw = torch.cat([p(pre + "res_conv.weight"), p(pre + "skip_conv.weight")], 0)[:, :, 0]
            rb = torch.cat([p(pre + "res_conv.bias"), p(pre + "skip_conv.bias")], 0)
            self.blocks.append((conv, rw.contiguous(), rb.view(-1, 1).contiguous()))

        self.out1_w, self.out1_b = p("renderer.output_conv1.weight")[:, :, 0], p("renderer.output_conv1.bias").view(-1, 1)
        self.out2_w, self.out2_b = p("renderer.output_conv2.weight")[:, :, 0], p("renderer.output_conv2.bias").view(-1, 1)
        self.rc = rc
        self.receptive_field = (1 + model_cfg["num_blocks"] * (k - 1) * (2 ** lpb - 1)
                                + 2 * (self.pe1.k - 1))

    @classmethod
    def load(cls, checkpoint_path, embedding_path, device="cpu", dtype=torch.float32):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        emb = torch.load(embedding_path, map_location="cpu", weights_only=False)
        want = emb.get("checkpoint")
        if want and Path(want).name != Path(checkpoint_path).name:
            print(f"warning: embedding was computed with {Path(want).name}, but you loaded "
                  f"{Path(checkpoint_path).name}; the sound may be wrong.")
        eng = cls(ckpt["model_state_dict"], ckpt["config"]["model"], emb["z"], device, dtype)
        eng.sample_rate = ckpt["config"]["data"]["sample_rate"]
        eng.name = emb.get("timbre", Path(embedding_path).stem)
        return eng

    def reset(self):
        for c in [self.pe1, self.pe2] + [b[0] for b in self.blocks]:
            c.reset()

    @torch.inference_mode()
    def process(self, x):
        """x: 1-D float tensor of N new dry samples -> 1-D tensor of N wet samples."""
        h = x.to(self.device, self.dtype).view(1, -1)
        h = F.relu(self.pe1(h))
        h = F.relu(self.pe2(h))
        skip_sum = 0.0
        for conv, rw, rb in self.blocks:
            s = torch.sigmoid_(conv(h))
            act = s[:self.rc].mul(2).sub_(1).mul_(s[self.rc:])
            rs = torch.addmm(rb, rw, act)
            h = (h + rs[:self.rc]) * SQRT_HALF
            skip_sum = skip_sum + rs[self.rc:]
        out = F.relu(skip_sum)
        out = F.relu(torch.addmm(self.out1_b, self.out1_w, out))
        out = torch.tanh(torch.addmm(self.out2_b, self.out2_w, out))
        return out.view(-1)
