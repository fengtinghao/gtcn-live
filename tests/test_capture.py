import numpy as np

from gtcn_live.capture import ReferenceCapture


def simulate(wet, latency, block=64):
    """Player hears wet and plays it back `latency` samples late (the dry take)."""
    cap = ReferenceCapture(wet, latency)
    heard = np.zeros(len(wet) + 2 * latency + 2 * block, dtype=np.float32)
    out = np.zeros(block, dtype=np.float32)
    while not cap.done:
        pos = cap.pos
        dry_in = heard[pos:pos + block].copy()
        cap.process(dry_in, out)
        heard[pos + latency:pos + latency + block] = out
    return cap


def test_dry_is_shifted_back_by_latency_so_it_lines_up_with_wet():
    wet = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    dry, got_wet = simulate(wet, latency=130).aligned()
    assert dry.shape == got_wet.shape == (1000,)
    assert np.array_equal(dry, wet)


def test_capture_finishes_after_wet_plus_latency_and_pads_silence():
    wet = np.ones(100, dtype=np.float32)
    cap = ReferenceCapture(wet, 30)
    out = np.empty(50, dtype=np.float32)
    played = []
    while not cap.done:
        cap.process(np.zeros(50, dtype=np.float32), out)
        played.append(out.copy())
    played = np.concatenate(played)
    assert cap.pos == 130
    assert np.all(played[:100] == 1) and np.all(played[100:] == 0)
