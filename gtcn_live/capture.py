"""Play a wet recording while recording the guitarist's dry playing along with it."""

import numpy as np


class ReferenceCapture:
    """Audio-callback logic: feeds out the wet recording, records the dry input.

    The player reacts to what they hear, so the dry signal arrives late by the
    interface round-trip latency; it is shifted back by that amount at the end.
    """

    def __init__(self, wet, latency_samples=0):
        self.wet = np.ascontiguousarray(wet, dtype=np.float32)
        self.latency = int(latency_samples)
        self.dry = np.zeros(len(self.wet) + self.latency, dtype=np.float32)
        self.pos = 0
        self.peak = 0.0
        self.xruns = 0

    @property
    def done(self):
        return self.pos >= len(self.dry)

    def process(self, dry_in, out):
        n = min(len(dry_in), len(self.dry) - self.pos)
        self.dry[self.pos:self.pos + n] = dry_in[:n]
        if n:
            self.peak = max(self.peak * 0.9, float(np.abs(dry_in[:n]).max()))
        out[:] = 0.0
        m = max(min(n, len(self.wet) - self.pos), 0)
        out[:m] = self.wet[self.pos:self.pos + m]
        self.pos += n

    def aligned(self):
        """(dry, wet) of equal length with the latency removed from the dry take."""
        return self.dry[self.latency:self.latency + len(self.wet)], self.wet
