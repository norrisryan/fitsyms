"""
Unit tests for fitsyms.sampling.

Tests verify distributional properties of each sampler using large sample
counts and statistical tolerance appropriate for 1M draws.
"""

import numpy as np
import pytest

from fitsyms.config import FehPriorConfig, InclPriorConfig, CompanionPriorConfig
from fitsyms.sampling import (
    sample_split_normal,
    sample_feh,
    feh_err_for_mist,
    sample_sini,
    sample_companion_mass,
    sample_eccentricity,
)

RNG = np.random.default_rng(20260501)
N = 500_000


class TestSplitNormal:
    def test_symmetric_is_gaussian(self):
        s = sample_split_normal(5.0, 1.0, 1.0, N, RNG)
        assert pytest.approx(s.mean(), abs=0.05) == 5.0
        assert pytest.approx(s.std(), abs=0.05) == 1.0

    def test_mode_at_mu(self):
        s = sample_split_normal(3.0, 0.5, 2.0, N, RNG)
        # The mode is mu; median shifts toward the wider side
        assert s.min() < 3.0
        assert s.max() > 3.0

    def test_asymmetry_direction(self):
        # sigma_hi > sigma_lo => median > mu
        s = sample_split_normal(0.0, 0.5, 2.0, N, RNG)
        assert np.median(s) > 0.0


class TestFehSamplers:
    def test_uniform_bounds(self):
        cfg = FehPriorConfig(kind="uniform", feh_lo=-0.5, feh_hi=+0.5)
        s = sample_feh(cfg, N, RNG)
        assert s.min() >= -0.5
        assert s.max() <= +0.5

    def test_truncated_gaussian_bounds(self):
        cfg = FehPriorConfig(kind="truncated_gaussian", mu=-0.54, sigma=0.10,
                             feh_lo=-1.0, feh_hi=+0.5)
        s = sample_feh(cfg, N, RNG)
        assert s.min() >= -1.0
        assert s.max() <= +0.5
        assert pytest.approx(s.mean(), abs=0.05) == -0.54

    def test_gaussian_mean(self):
        cfg = FehPriorConfig(kind="gaussian", mu=-0.17, sigma=0.08)
        s = sample_feh(cfg, N, RNG)
        assert pytest.approx(s.mean(), abs=0.01) == -0.17

    def test_unknown_kind_raises(self):
        cfg = FehPriorConfig(kind="unknown")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown feh_prior"):
            sample_feh(cfg, 10, RNG)

    def test_feh_err_uniform(self):
        cfg = FehPriorConfig(kind="uniform", feh_lo=-0.4, feh_hi=+0.4)
        err = feh_err_for_mist(cfg)
        import math
        expected = 0.8 / math.sqrt(12)
        assert pytest.approx(err, rel=1e-6) == expected

    def test_feh_err_gaussian(self):
        cfg = FehPriorConfig(kind="truncated_gaussian", mu=0.0, sigma=0.10)
        assert feh_err_for_mist(cfg) == 0.10


class TestSiniSamplers:
    def test_isotropic_truncated_bounds(self):
        cfg = InclPriorConfig(kind="isotropic_truncated",
                              incl_min=50.0, incl_max=90.0)
        s = sample_sini(cfg, N, RNG)
        import math
        sin_min = math.sin(math.radians(50.0))
        assert np.all(s >= sin_min - 1e-10)
        assert np.all(s <= 1.0 + 1e-10)

    def test_isotropic_full_range(self):
        cfg = InclPriorConfig(kind="isotropic")
        s = sample_sini(cfg, N, RNG)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_gaussian_deg_near_mode(self):
        cfg = InclPriorConfig(kind="gaussian_deg", incl_deg=90.0, incl_err_deg=1.0)
        s = sample_sini(cfg, N, RNG)
        # sin(90) = 1; tight Gaussian => most samples near 1
        assert np.median(s) > 0.99

    def test_unknown_kind_raises(self):
        cfg = InclPriorConfig(kind="bad_kind")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown incl_prior"):
            sample_sini(cfg, 10, RNG)


class TestCompanionMassSamplers:
    def test_flat_bounds(self):
        cfg = CompanionPriorConfig(kind="flat", lo=0.35, hi=0.60)
        s = sample_companion_mass(cfg, N, RNG)
        assert s.min() >= 0.35
        assert s.max() <= 0.60

    def test_gaussian_truncated(self):
        cfg = CompanionPriorConfig(kind="gaussian", lo=0.3, hi=1.44,
                                   mu=0.6, sigma=0.1)
        s = sample_companion_mass(cfg, N, RNG)
        assert s.min() >= 0.3
        assert s.max() <= 1.44
        assert pytest.approx(s.mean(), abs=0.02) == 0.6


class TestEccentricitySampler:
    def test_circular_is_delta(self):
        s = sample_eccentricity(0.0, 0.0, N, RNG)
        assert np.all(s == 0.0)

    def test_truncated_at_zero(self):
        s = sample_eccentricity(0.05, 0.02, N, RNG)
        assert np.all(s >= 0.0)
        assert np.all(s < 0.99)

    def test_mean_near_centre(self):
        s = sample_eccentricity(0.1, 0.01, N, RNG)
        assert pytest.approx(s.mean(), abs=0.01) == 0.1
