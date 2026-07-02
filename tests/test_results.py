"""
Unit tests for InferenceResult save/load round-trip.

Tests verify that all scalar summaries and posterior arrays survive an
HDF5 serialisation/deserialisation cycle without precision loss beyond
float32 tolerance.
"""

import json

import numpy as np
import pytest

from fitsyms.results import InferenceResult


def _make_result(has_orbit: bool = True, n: int = 1000) -> InferenceResult:
    """Build a minimal InferenceResult with synthetic posterior arrays."""
    rng = np.random.default_rng(42)
    mg = rng.normal(1.5, 0.2, n).astype(np.float64)
    R  = rng.normal(80.0, 5.0, n).astype(np.float64)

    base = dict(
        name="Test Star",
        has_orbit=has_orbit,
        n_samples=n,
        k_neighbors=16,
        logT_obs=3.56,
        logT_err=0.014,
        logL_obs=2.85,
        logL_err=0.08,
        R_central=82.0,
        R_err_central=4.0,
        L_central=5000.0,
        L_err_central=400.0,
        mg_med=1.52, mg_pu=0.18, mg_pl=0.16,
        R_med=80.0,  R_pu=5.0,   R_pl=4.8,
        L_med=4900., L_pu=350.,  L_pl=320.,
        chi2_mist_med=1.2,
        n_unique_nodes=800,
        ess=float(n),
        mg_post=mg,
        R_post=R,
        feh_post=rng.normal(-0.2, 0.1, n),
        dist_post=rng.normal(300.0, 10.0, n),
        teff_post=rng.normal(3600.0, 100.0, n),
    )

    if has_orbit:
        m2 = rng.normal(0.45, 0.05, n)
        ff = rng.normal(0.75, 0.05, n)
        base.update(dict(
            m2_med=0.45,  m2_pu=0.05,  m2_pl=0.04,
            ff_med=0.75,  ff_pu=0.05,  ff_pl=0.05,
            ffp_med=0.78, ffp_pu=0.05, ffp_pl=0.05,
            rl_med=110.,  rl_pu=8.,    rl_pl=7.,
            a_med=180.,   a_pu=12.,    a_pl=10.,
            sin_i_med=0.95,
            m2_post=m2,
            ff_post=ff,
            ffp_post=ff * 1.03,
            sini_post=rng.uniform(0.85, 1.0, n),
        ))
    else:
        base.update(dict(
            mg_adopted=2.78,
            mg_adopted_err=0.63,
            mg_adopted_cite="Ilkiewicz+2022",
        ))

    return InferenceResult(**base)


class TestRoundTrip:
    def test_scalar_round_trip(self, tmp_path):
        r = _make_result(has_orbit=True)
        r.save(tmp_path / "test.h5")
        r2 = InferenceResult.load(tmp_path / "test.h5")
        assert r2.name == r.name
        assert r2.has_orbit == r.has_orbit
        assert pytest.approx(r2.mg_med) == r.mg_med
        assert pytest.approx(r2.ff_med) == r.ff_med
        assert pytest.approx(r2.ess) == r.ess

    def test_posterior_round_trip_float32_tolerance(self, tmp_path):
        r = _make_result(has_orbit=True)
        r.save(tmp_path / "test.h5")
        r2 = InferenceResult.load(tmp_path / "test.h5")
        # Float32 precision: ~7 significant digits
        np.testing.assert_allclose(r2.mg_post, r.mg_post, rtol=1e-5)
        np.testing.assert_allclose(r2.ff_post, r.ff_post, rtol=1e-5)

    def test_no_orbit_round_trip(self, tmp_path):
        r = _make_result(has_orbit=False)
        r.save(tmp_path / "test_noorbit.h5")
        r2 = InferenceResult.load(tmp_path / "test_noorbit.h5")
        assert r2.has_orbit is False
        assert r2.ff_med is None
        assert r2.m2_med is None
        assert pytest.approx(r2.mg_adopted) == 2.78
        assert r2.mg_adopted_cite == "Ilkiewicz+2022"

    def test_load_scalars_only(self, tmp_path):
        r = _make_result(has_orbit=True)
        r.save(tmp_path / "test.h5")
        r2 = InferenceResult.load(tmp_path / "test.h5", load_posteriors=False)
        assert r2.mg_post is None
        assert pytest.approx(r2.mg_med) == r.mg_med

    def test_json_written_alongside_h5(self, tmp_path):
        r = _make_result(has_orbit=True)
        r.save(tmp_path / "test.h5")
        json_path = tmp_path / "test.json"
        assert json_path.exists()
        with open(json_path) as fh:
            d = json.load(fh)
        assert d["name"] == "Test Star"
        assert pytest.approx(d["mg_med"]) == r.mg_med
        # Arrays must NOT appear in JSON
        assert "mg_post" not in d

    def test_h5_extension_added_if_missing(self, tmp_path):
        r = _make_result()
        r.save(tmp_path / "test")   # no extension
        assert (tmp_path / "test.h5").exists()

    def test_summary_line_has_orbit(self):
        r = _make_result(has_orbit=True)
        line = r.summary_line()
        assert "Mg=" in line
        assert "M2=" in line
        assert "f=" in line

    def test_summary_line_no_orbit(self):
        r = _make_result(has_orbit=False)
        line = r.summary_line()
        assert "MIST only" in line
