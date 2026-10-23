"""Create effect embeddings from aligned dry/wet reference recordings."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def read_reference_pair(dry_path, wet_path, sample_rate):
    dry, dry_sr = sf.read(dry_path, dtype="float32", always_2d=True)
    wet, wet_sr = sf.read(wet_path, dtype="float32", always_2d=True)

    if dry_sr != sample_rate or wet_sr != sample_rate:
        raise ValueError(
            f"Both reference WAVs must be {sample_rate} Hz "
            f"(dry is {dry_sr} Hz, wet is {wet_sr} Hz)."
        )
    if dry.shape[0] != wet.shape[0]:
        raise ValueError(
            f"Dry and wet reference WAVs must have the same number of samples "
            f"(dry has {dry.shape[0]}, wet has {wet.shape[0]})."
        )
    if not np.isfinite(dry).all() or not np.isfinite(wet).all():
        raise ValueError("Reference WAVs must contain only finite audio samples.")

    return dry.mean(axis=1), wet.mean(axis=1)


@torch.inference_mode()
def encode_reference(model, dry, wet, sample_rate, reference_seconds, trim_fraction=0.1):
    reference_length = round(sample_rate * reference_seconds)
    if reference_length <= 0:
        raise ValueError("Model reference duration must be positive.")
    if len(dry) < reference_length:
        raise ValueError(
            f"Reference recordings must be at least {reference_seconds:g} seconds "
            f"({reference_length} samples) long."
        )
    if not 0 <= trim_fraction < 1:
        raise ValueError("trim_fraction must be in the range [0, 1).")

    device = next(model.parameters()).device
    embeddings = []
    for start in range(0, len(dry) - reference_length + 1, reference_length):
        dry_ref = torch.from_numpy(dry[start:start + reference_length]).to(device).view(1, 1, -1)
        wet_ref = torch.from_numpy(wet[start:start + reference_length]).to(device).view(1, 1, -1)
        embeddings.append(model.encode_effect(dry_ref, wet_ref).squeeze(0).cpu())

    values = torch.stack(embeddings)
    center = values.median(dim=0).values
    distances = torch.linalg.vector_norm(values - center, dim=1)
    keep_count = max(1, int(len(values) * (1.0 - trim_fraction)))
    keep = torch.argsort(distances)[:keep_count]
    z = values[keep].mean(dim=0)
    return z, {
        "num_references": len(embeddings),
        "num_inliers": int(len(keep)),
        "reference_seconds": reference_seconds,
    }


def load_model(checkpoint_path, gtcn_source):
    source = Path(gtcn_source).resolve()
    if not (source / "model.py").is_file():
        raise ValueError(f"Could not find model.py under gTCN source directory: {source}")

    import sys

    sys.path.insert(0, str(source))
    from model import GTCNModel

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = GTCNModel.from_config(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint_path


def save_embedding(model, config, checkpoint_path, dry, wet, output_path, name, metadata=None):
    sample_rate = config["data"]["sample_rate"]
    reference_seconds = config["data"].get("reference_seconds", 2.0)
    z, stats = encode_reference(model, dry, wet, sample_rate, reference_seconds)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    effect_name = name or output_path.stem.removesuffix("_embeddings")
    torch.save(
        {
            "checkpoint": str(checkpoint_path),
            "timbre": effect_name,
            "z": z,
            "metadata": {**stats, **(metadata or {})},
        },
        output_path,
    )
    return effect_name, stats


def create_embedding(checkpoint_path, gtcn_source, dry_path, wet_path, output_path, name=None):
    model, config, checkpoint_path = load_model(checkpoint_path, gtcn_source)
    dry, wet = read_reference_pair(dry_path, wet_path, config["data"]["sample_rate"])
    return save_embedding(
        model, config, checkpoint_path, dry, wet, output_path, name,
        {"dry_reference": Path(dry_path).name, "wet_reference": Path(wet_path).name},
    )
