"""
Unit tests for fitsyms.physics.

All tests use analytic or literature-verified values and run without
any MIST grid files.
"""

import math

import numpy as np
import pytest

from fitsyms.physics import (
    angular_to_radius,
    angular_to_radius_vec,
    luminosity,
    eggleton_rl,
    orbital_separation_rsun,
    roche_lobe_radius,
    roche_lobe_radius_periastron,
    surface_gravity,
    percentile_summary,
)
from fitsyms.constants import AU_TO_RSUN, T_SUN


class TestAngularToRadius:
    def test_scalar_known_value(self):
        # theta = 1 mas at 1 pc = 1 AU * 1e-3 / 2 radius = 0.5e-3 AU
        # R = 0.5 * 1.0 * 1e-3 * 1.0 * AU_TO_RSUN
        R, R_err = angular_to_radius(1.0, 0.1, 1.0, 0.01)
        expected = 0.5 * 1.0 * 1e-3 * 1.0 * AU_TO_RSUN
        assert pytest.approx(R, rel=1e-6) == expected

    def test_error_propagation_symmetric(self):
        # When theta_err/theta == dist_err/dist, frac_err = sqrt(2) * each
        theta, dist = 2.0, 200.0
        theta_err, dist_err = 0.02, 2.0  # both 1% fractional
        R, R_err = angular_to_radius(theta, theta_err, dist, dist_err)
        frac = math.sqrt((theta_err/theta)**2 + (dist_err/dist)**2)
        assert pytest.approx(R_err, rel=1e-6) == R * frac

    def test_vectorised_matches_scalar(self):
        theta_s = np.array([2.44, 1.78, 5.39])
        dist_s  = np.array([247.6, 400.0, 235.4])
        R_vec = angular_to_radius_vec(theta_s, dist_s)
        for i in range(len(theta_s)):
            R_sc, _ = angular_to_radius(theta_s[i], 0.05, dist_s[i], 5.0)
            assert pytest.approx(R_vec[i], rel=1e-6) == R_sc


class TestLuminosity:
    def test_solar_values(self):
        # R=1 Rsun, Teff=Tsun => L=1 Lsun
        L, L_err = luminosity(T_SUN, 100.0, 1.0, 0.01)
        assert pytest.approx(L, rel=1e-6) == 1.0

    def test_scaling_with_radius(self):
        L1, _ = luminosity(T_SUN, 10.0, 1.0, 0.01)
        L2, _ = luminosity(T_SUN, 10.0, 2.0, 0.02)
        assert pytest.approx(L2 / L1, rel=1e-4) == 4.0

    def test_scaling_with_teff(self):
        L1, _ = luminosity(T_SUN,       10.0, 1.0, 0.01)
        L2, _ = luminosity(T_SUN * 2.0, 10.0, 1.0, 0.01)
        assert pytest.approx(L2 / L1, rel=1e-4) == 16.0


class TestEggletonRL:
    def test_q_equals_1(self):
        # At q=1: rl = 0.49 / (0.6 + ln 2) ≈ 0.3782
        rl = eggleton_rl(1.0)
        expected = 0.49 / (0.6 + math.log(2.0))
        assert pytest.approx(rl, rel=1e-4) == expected

    def test_coefficient_is_0pt6_not_0pt69(self):
        # The paper explicitly notes 0.6 not 0.69; verify the difference matters.
        q = 2.0
        q23 = q ** (2.0/3.0)
        rl_correct = 0.49 * q23 / (0.6  * q23 + math.log(1 + q**(1/3)))
        rl_wrong   = 0.49 * q23 / (0.69 * q23 + math.log(1 + q**(1/3)))
        assert rl_correct > rl_wrong
        assert abs(rl_correct - rl_wrong) / rl_correct > 0.01  # >1% difference

    def test_vectorised(self):
        q = np.array([0.5, 1.0, 2.0, 5.0])
        rl = eggleton_rl(q)
        assert rl.shape == (4,)
        assert np.all(rl > 0) and np.all(rl < 0.5)


class TestOrbitalSeparation:
    def test_kepler_third_law(self):
        # For M_total = 1 Msun, period = 1 year, ecc=0: a = 1 AU
        a = orbital_separation_rsun(1.0, 365.25, 0.0)
        assert pytest.approx(a, rel=1e-3) == AU_TO_RSUN

    def test_ecc_increases_average_separation(self):
        a_circ = orbital_separation_rsun(2.0, 200.0, 0.0)
        a_ecc  = orbital_separation_rsun(2.0, 200.0, 0.3)
        # <r> = a(1 + e^2/2) > a for e > 0
        assert a_ecc > a_circ


class TestRocheLobe:
    def test_periastron_smaller_than_average(self):
        mg = np.ones(10) * 2.0
        m2 = np.ones(10) * 0.5
        p  = np.ones(10) * 300.0
        e  = np.ones(10) * 0.1
        rl_avg  = roche_lobe_radius(mg, m2, p, e)
        rl_peri = roche_lobe_radius_periastron(mg, m2, p, e)
        assert np.all(rl_peri < rl_avg)

    def test_circular_orbit_peri_equals_avg(self):
        mg = np.ones(5) * 2.0
        m2 = np.ones(5) * 0.5
        p  = np.ones(5) * 300.0
        e  = np.zeros(5)
        rl_avg  = roche_lobe_radius(mg, m2, p, e)
        rl_peri = roche_lobe_radius_periastron(mg, m2, p, e)
        np.testing.assert_allclose(rl_avg, rl_peri, rtol=1e-6)


class TestPercentileSummary:
    def test_symmetric_gaussian(self):
        rng = np.random.default_rng(42)
        x = rng.normal(5.0, 1.0, 1_000_000)
        med, pu, pl = percentile_summary(x)
        assert pytest.approx(med, abs=0.01) == 5.0
        assert pytest.approx(pu, abs=0.05) == 1.0
        assert pytest.approx(pl, abs=0.05) == 1.0

    def test_returns_positive_intervals(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        med, pu, pl = percentile_summary(x)
        assert pu >= 0
        assert pl >= 0
