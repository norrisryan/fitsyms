"""
Unit tests for fitsyms.config (YAML loading and validation).

These tests use a minimal in-memory YAML string to avoid depending on
the actual config files or the MIST grid.
"""

import textwrap
from pathlib import Path

import pytest
import yaml

from fitsyms.config import TargetConfig, _validate_feh_prior, _validate_companion_prior
from fitsyms.config import FehPriorConfig, CompanionPriorConfig


MINIMAL_YAML = textwrap.dedent("""\
    name: "Test Star"
    mist_dir: "/tmp/mist"
    gardner_factor: 1.0054

    angular_diameter:
      theta_mas: 2.0
      err_lo: 0.05
      err_hi: 0.05

    distance:
      dist_pc: 300.0
      err_lo: 10.0
      err_hi: 10.0
      source: "Test source"

    spectroscopy:
      teff: 3600
      teff_err: 100
      av: 0.1
      av_err: 0.05
      feh_starfish: 0.0
      feh_starfish_err: 0.25

    feh_prior:
      kind: truncated_gaussian
      mu: -0.20
      sigma: 0.10
      feh_lo: -1.0
      feh_hi: +0.5

    companion_prior:
      kind: flat
      lo: 0.30
      hi: 1.44

    incl_prior:
      kind: isotropic_truncated
      incl_min: 50.0
      incl_max: 90.0

    orbit:
      has_orbit: true
      fm: 0.037
      fm_err: 0.003
      period_d: 596.21
      period_err_d: 0.19
      ecc: 0.088
      ecc_err: 0.023

    inference:
      n_samples: 100000
      k_neighbors: 16
      rng_seed: 42
""")


def _load_yaml_str(yaml_str: str, tmp_path: Path) -> TargetConfig:
    """Write yaml_str to a temp file and load it."""
    p = tmp_path / "test.yaml"
    p.write_text(yaml_str)
    return TargetConfig.from_yaml(p)


class TestMinimalConfig:
    def test_loads_without_error(self, tmp_path):
        cfg = _load_yaml_str(MINIMAL_YAML, tmp_path)
        assert cfg.name == "Test Star"

    def test_gardner_correction_applied(self, tmp_path):
        cfg = _load_yaml_str(MINIMAL_YAML, tmp_path)
        expected = 2.0 / 1.0054
        assert pytest.approx(cfg.theta_corrected, rel=1e-6) == expected

    def test_inference_defaults_overridden(self, tmp_path):
        cfg = _load_yaml_str(MINIMAL_YAML, tmp_path)
        assert cfg.inference.n_samples == 100_000
        assert cfg.inference.k_neighbors == 16
        assert cfg.inference.rng_seed == 42

    def test_orbit_fields(self, tmp_path):
        cfg = _load_yaml_str(MINIMAL_YAML, tmp_path)
        assert cfg.orbit.has_orbit is True
        assert pytest.approx(cfg.orbit.fm) == 0.037
        assert pytest.approx(cfg.orbit.ecc) == 0.088

    def test_feh_prior_fields(self, tmp_path):
        cfg = _load_yaml_str(MINIMAL_YAML, tmp_path)
        assert cfg.feh_prior.kind == "truncated_gaussian"
        assert pytest.approx(cfg.feh_prior.mu) == -0.20
        assert pytest.approx(cfg.feh_prior.sigma) == 0.10

    def test_theta_err_avg(self, tmp_path):
        cfg = _load_yaml_str(MINIMAL_YAML, tmp_path)
        assert pytest.approx(cfg.theta_err_avg_corrected) == 0.05 / 1.0054

    def test_missing_required_key_raises(self, tmp_path):
        raw = yaml.safe_load(MINIMAL_YAML)
        del raw["orbit"]
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(raw))
        with pytest.raises(ValueError, match="missing required keys"):
            TargetConfig.from_yaml(p)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            TargetConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_no_orbit_config(self, tmp_path):
        raw = yaml.safe_load(MINIMAL_YAML)
        raw["orbit"]["has_orbit"] = False
        p = tmp_path / "noorbit.yaml"
        p.write_text(yaml.dump(raw))
        cfg = TargetConfig.from_yaml(p)
        assert cfg.orbit.has_orbit is False

    def test_mg_adopted_optional(self, tmp_path):
        cfg = _load_yaml_str(MINIMAL_YAML, tmp_path)
        assert cfg.mg_adopted is None
        raw = yaml.safe_load(MINIMAL_YAML)
        raw["mg_adopted"] = 2.78
        raw["mg_adopted_err"] = 0.63
        p = tmp_path / "with_adopted.yaml"
        p.write_text(yaml.dump(raw))
        cfg2 = TargetConfig.from_yaml(p)
        assert pytest.approx(cfg2.mg_adopted) == 2.78


class TestValidation:
    def test_feh_prior_bad_kind(self):
        cfg = FehPriorConfig(kind="bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown feh_prior kind"):
            _validate_feh_prior(cfg)

    def test_feh_prior_lo_ge_hi(self):
        cfg = FehPriorConfig(kind="uniform", feh_lo=0.5, feh_hi=-0.5)
        with pytest.raises(ValueError, match="feh_lo.*must be"):
            _validate_feh_prior(cfg)

    def test_feh_prior_zero_sigma(self):
        cfg = FehPriorConfig(kind="truncated_gaussian", mu=0.0, sigma=0.0)
        with pytest.raises(ValueError, match="sigma must be"):
            _validate_feh_prior(cfg)

    def test_companion_prior_lo_ge_hi(self):
        cfg = CompanionPriorConfig(kind="flat", lo=1.0, hi=0.5)
        with pytest.raises(ValueError, match="lo.*must be"):
            _validate_companion_prior(cfg)

    def test_companion_prior_gaussian_missing_mu(self):
        cfg = CompanionPriorConfig(kind="gaussian", lo=0.3, hi=1.44,
                                   mu=None, sigma=0.1)
        with pytest.raises(ValueError, match="requires mu and sigma"):
            _validate_companion_prior(cfg)
