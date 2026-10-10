import numpy as np

from exptools.freeze import fingerprint


def test_fingerprint_is_pinned():
    # The committed Phase 4 and 4c fingerprints were made with exactly this function;
    # if this value changes, every frozen model would fail its refit check.
    assert fingerprint(np.arange(7) / 3) == "03b9125a0ac1981e"


def test_fingerprint_detects_real_changes():
    values = np.linspace(0, 1, 1_000)
    assert fingerprint(values.copy()) == fingerprint(values)
    assert fingerprint(values + 1e-6) != fingerprint(values)


def test_rounding_does_not_absorb_noise_at_a_rounding_boundary():
    # A value on a 10-decimal boundary flips under a 1e-13 nudge, so the freeze check
    # needs bit-for-bit deterministic refits, not just rounding.
    on_boundary = np.array([0.12345678905])
    assert fingerprint(on_boundary + 1e-13) != fingerprint(on_boundary - 1e-13)


def test_fingerprint_depends_on_dtype():
    # Why the function never casts: 32- and 64-bit copies of the same numbers hash differently.
    values = np.arange(5) / 4
    assert fingerprint(values.astype(np.float32)) != fingerprint(values)
