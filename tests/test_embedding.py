import numpy as np
import pytest
import soundfile as sf
import torch

from gtcn_live.embedding import encode_reference, read_reference_pair


class MeanEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode_effect(self, dry, wet):
        return torch.stack((dry.mean(), wet.mean())).view(1, -1)


def test_read_reference_pair_checks_sample_rate_and_alignment(tmp_path):
    dry_path = tmp_path / "dry.wav"
    wet_path = tmp_path / "wet.wav"
    sf.write(dry_path, np.zeros(16, dtype=np.float32), 8, subtype="FLOAT")
    sf.write(wet_path, np.ones(16, dtype=np.float32), 8, subtype="FLOAT")

    dry, wet = read_reference_pair(dry_path, wet_path, 8)
    assert dry.shape == wet.shape == (16,)
    assert np.all(dry == 0)
    assert np.allclose(wet, 1)

    with pytest.raises(ValueError, match="must be 16 Hz"):
        read_reference_pair(dry_path, wet_path, 16)

    sf.write(wet_path, np.ones(8, dtype=np.float32), 8, subtype="FLOAT")
    with pytest.raises(ValueError, match="same number of samples"):
        read_reference_pair(dry_path, wet_path, 8)


def test_encode_reference_averages_reference_windows():
    dry = np.arange(8, dtype=np.float32)
    wet = dry * 2

    z, stats = encode_reference(MeanEncoder(), dry, wet, 4, 0.5, trim_fraction=0)

    assert torch.allclose(z, torch.tensor([3.5, 7.0]))
    assert stats == {
        "num_references": 4,
        "num_inliers": 4,
        "reference_seconds": 0.5,
    }


def test_encode_reference_requires_training_reference_duration():
    with pytest.raises(ValueError, match="at least 1 seconds"):
        encode_reference(MeanEncoder(), np.zeros(7, dtype=np.float32),
                         np.zeros(7, dtype=np.float32), 8, 1)
