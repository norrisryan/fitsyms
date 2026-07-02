"""
Gaudin et al. sensitivity analysis runs.

This script reproduces all sensitivity tables from the paper using the
FITSYMS public API.  It is NOT part of the installable package; it lives
in paper/ to document the exact prior choices tested for the referee
response and the paper's appendix tables.

Usage
-----
    python paper/run_sensitivity.py

Output is printed to stdout. Redirect to a file for record-keeping::

    python paper/run_sensitivity.py | tee paper/sensitivity_output.txt

Dependencies: fitsyms (installed), numpy
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from fitsyms import MISTGrid, TargetConfig, run_inference
from fitsyms.config import FehPriorConfig, InclPriorConfig, CompanionPriorConfig
from fitsyms.constants import FEH_PRIOR_LO, FEH_PRIOR_HI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).parent.parent / "config"
N_SENS = 1_000_000   # sample count for all sensitivity runs (not N_MAIN)

CONFIGS = {
    "V1472 Aql": TargetConfig.from_yaml(CONFIG_DIR / "v1472aql.yaml"),
    "EG And":    TargetConfig.from_yaml(CONFIG_DIR / "egand.yaml"),
    "BD Cam":    TargetConfig.from_yaml(CONFIG_DIR / "bdcam.yaml"),
    "SU Lyn":    TargetConfig.from_yaml(CONFIG_DIR / "sulyn.yaml"),
}

# Load MIST grid once (all targets share the same directory).
_mist_dir = next(iter(CONFIGS.values())).mist_dir
print(f"\nLoading MIST grid: {_mist_dir}")
GRID = MISTGrid(_mist_dir)


def run(cfg: TargetConfig, label: str = "") -> dict:
    """Run inference and return the result dict for table printing."""
    r = run_inference(cfg, GRID, n_samples=N_SENS)
    return r


def _header(title: str, width: int = 82) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def _fmt(r, name: str) -> str:
    if r.has_orbit:
        return (
            f"{name:<26} "
            f"Mg={r.mg_med:.2f}+{r.mg_pu:.2f}-{r.mg_pl:.2f}  "
            f"M2={r.m2_med:.2f}+{r.m2_pu:.2f}-{r.m2_pl:.2f}  "
            f"f={r.ff_med:.3f}+{r.ff_pu:.3f}-{r.ff_pl:.3f}  "
            f"ESS={r.ess:.0f}"
        )
    return (
        f"{name:<26} "
        f"Mg={r.mg_med:.2f}+{r.mg_pu:.2f}-{r.mg_pl:.2f} (MIST only)  "
        f"ESS={r.ess:.0f}"
    )


# ---------------------------------------------------------------------------
# 1. [Fe/H] prior sensitivity — all four targets
# ---------------------------------------------------------------------------

FEH_CASES: dict[str, list[tuple[str, FehPriorConfig]]] = {
    "V1472 Aql": [
        ("Hayden+2015 disc MDF (main)",
         FehPriorConfig(kind="truncated_gaussian", mu=+0.02, sigma=0.20,
                        feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI,
                        cite="Hayden+2015")),
        ("uniform [-1.0,+0.5]",
         FehPriorConfig(kind="uniform", feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI)),
        ("Starfish Z_free (+0.015, 0.259)",
         FehPriorConfig(kind="truncated_gaussian", mu=+0.015, sigma=0.259,
                        feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI)),
    ],
    "EG And": [
        ("Galan+2023 (-0.54, 0.10) (main)",
         FehPriorConfig(kind="truncated_gaussian", mu=-0.54, sigma=0.10,
                        feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI,
                        cite="Galan+2023")),
        ("uniform [-1.0,+0.5]",
         FehPriorConfig(kind="uniform", feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI)),
        ("Starfish Z_free (-0.054, 0.249)",
         FehPriorConfig(kind="truncated_gaussian", mu=-0.054, sigma=0.249,
                        feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI)),
        ("Z=-0.17 Gauss(-0.17, 0.08)",
         FehPriorConfig(kind="gaussian", mu=-0.17, sigma=0.08)),
    ],
    "BD Cam": [
        ("Charbonnel+2020 (-0.20, 0.10) (main)",
         FehPriorConfig(kind="truncated_gaussian", mu=-0.20, sigma=0.10,
                        feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI,
                        cite="Charbonnel+2020")),
        ("uniform [-1.0,+0.5]",
         FehPriorConfig(kind="uniform", feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI)),
        ("Starfish Z_free (-0.101, 0.242)",
         FehPriorConfig(kind="truncated_gaussian", mu=-0.101, sigma=0.242,
                        feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI)),
    ],
    "SU Lyn": [
        ("Hayden+2015 disc MDF (main)",
         FehPriorConfig(kind="truncated_gaussian", mu=+0.02, sigma=0.20,
                        feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI)),
        ("uniform [-1.0,+0.5]",
         FehPriorConfig(kind="uniform", feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI)),
        ("Starfish Z_free (+0.337, 0.123)",
         FehPriorConfig(kind="truncated_gaussian", mu=+0.337, sigma=0.123,
                        feh_lo=FEH_PRIOR_LO, feh_hi=FEH_PRIOR_HI)),
    ],
}

_header("[Fe/H] PRIOR SENSITIVITY — ALL TARGETS")

for name, cases in FEH_CASES.items():
    print(f"\n--- {name} ---")
    f_vals, mg_vals = [], []
    for label, feh_cfg in cases:
        cfg = copy.deepcopy(CONFIGS[name])
        cfg.feh_prior = feh_cfg
        r = run(cfg)
        print(_fmt(r, label))
        mg_vals.append(r.mg_med)
        if r.has_orbit:
            f_vals.append(r.ff_med)
    mg_spread = max(mg_vals) - min(mg_vals)
    f_spread  = (max(f_vals) - min(f_vals)) if f_vals else None
    mg_verdict = ("INSENSITIVE" if mg_spread < 0.20
                  else "MODERATE" if mg_spread < 0.50 else "SENSITIVE")
    f_verdict  = ("INSENSITIVE" if f_spread is not None and f_spread < 0.05
                  else "MODERATE" if f_spread is not None and f_spread < 0.10
                  else "SENSITIVE" if f_spread is not None else "N/A")
    print(f"  Mg spread: {mg_spread:.2f} Msun  => {mg_verdict}")
    if f_spread is not None:
        print(f"  f  spread: {f_spread:.3f}        => {f_verdict}")


# ---------------------------------------------------------------------------
# 2. Inclination prior sensitivity — V1472 Aql
# ---------------------------------------------------------------------------

_header("INCLINATION PRIOR SENSITIVITY — V1472 Aql")

INCL_CASES = [
    ("isotropic_truncated [50,90] (main)",
     InclPriorConfig(kind="isotropic_truncated", incl_min=50.0, incl_max=90.0)),
    ("uniform_deg [50,90] (v4 behaviour)",
     InclPriorConfig(kind="uniform_deg", incl_min=50.0, incl_max=90.0)),
    ("isotropic_truncated [70,90]",
     InclPriorConfig(kind="isotropic_truncated", incl_min=70.0, incl_max=90.0)),
    ("isotropic_truncated [50,70]",
     InclPriorConfig(kind="isotropic_truncated", incl_min=50.0, incl_max=70.0)),
]

f_vals_incl = []
for label, incl_cfg in INCL_CASES:
    cfg = copy.deepcopy(CONFIGS["V1472 Aql"])
    cfg.incl_prior = incl_cfg
    r = run(cfg)
    print(_fmt(r, label))
    f_vals_incl.append(r.ff_med)

f_nom = f_vals_incl[0]
print(f"\nShift in f relative to main (isotropic_truncated [50,90]):")
for (label, _), fv in zip(INCL_CASES, f_vals_incl):
    print(f"  {label:<42}  delta_f = {fv - f_nom:+.3f}")


# ---------------------------------------------------------------------------
# 3. EG And WD prior sensitivity
# ---------------------------------------------------------------------------

_header("EG And — WD PRIOR SENSITIVITY")

WD_CASES = [
    ("informed [0.35,0.55]",
     CompanionPriorConfig(kind="flat", lo=0.35, hi=0.55)),
    ("adopted  [0.35,0.60] (main)",
     CompanionPriorConfig(kind="flat", lo=0.35, hi=0.60)),
    ("broad    [0.30,1.44]",
     CompanionPriorConfig(kind="flat", lo=0.30, hi=1.44)),
    ("low-floor[0.20,1.44]",
     CompanionPriorConfig(kind="flat", lo=0.20, hi=1.44)),
]

for label, wd_cfg in WD_CASES:
    cfg = copy.deepcopy(CONFIGS["EG And"])
    cfg.companion_prior = wd_cfg
    r = run(cfg)
    print(_fmt(r, label))


# ---------------------------------------------------------------------------
# 4. EG And WD-prior × distance sensitivity
# ---------------------------------------------------------------------------

_header("EG And — WD PRIOR × DISTANCE SENSITIVITY")

DIST_CASES = [
    ("adopted 400 pc",  400.0,  20.0,  20.0),
    ("Gaia    593 pc",  593.52, 13.35, 11.87),
]

print(f"\n{'distance':<16} {'WD prior':<26} {'Mg':>18} {'M2':>18} {'f':>16}  ESS")
print("-" * 100)

for dlabel, dpc, elo, ehi in DIST_CASES:
    for wlabel, wd_cfg in WD_CASES:
        cfg = copy.deepcopy(CONFIGS["EG And"])
        cfg.distance.dist_pc = dpc
        cfg.distance.err_lo  = elo
        cfg.distance.err_hi  = ehi
        cfg.companion_prior  = wd_cfg
        r = run(cfg)
        print(
            f"{dlabel:<16} {wlabel:<26} "
            f"Mg={r.mg_med:.2f}+{r.mg_pu:.2f}-{r.mg_pl:.2f}  "
            f"M2={r.m2_med:.2f}+{r.m2_pu:.2f}-{r.m2_pl:.2f}  "
            f"f={r.ff_med:.3f}+{r.ff_pu:.3f}-{r.ff_pl:.3f}  "
            f"ESS={r.ess:.0f}"
        )
    print()


# ---------------------------------------------------------------------------
# 5. Grid discretisation diagnostic (k=1 vs k=16)
# ---------------------------------------------------------------------------

_header("GRID DISCRETISATION — k=1 (snap) vs k=16 (kernel-smoothed)")
print(f"\n{'Target':<12} {'Mg (k=1)':>22} {'Mg (k=16)':>22} {'dMg':>8} {'chi2_med':>10}")
print("-" * 80)

for name in ["V1472 Aql", "EG And", "BD Cam"]:
    cfg = CONFIGS[name]
    r1  = run_inference(cfg, GRID, n_samples=N_SENS, k_neighbors=1)
    r16 = run_inference(cfg, GRID, n_samples=N_SENS, k_neighbors=16)
    dmg = r16.mg_med - r1.mg_med
    flag = "" if abs(dmg) < 0.05 else "  <-- CHECK"
    print(
        f"{name:<12} "
        f"{r1.mg_med:.2f}+{r1.mg_pu:.2f}-{r1.mg_pl:.2f}          "
        f"{r16.mg_med:.2f}+{r16.mg_pu:.2f}-{r16.mg_pl:.2f}          "
        f"{dmg:>+6.3f} {r1.chi2_mist_med:>10.2f}{flag}"
    )


# ---------------------------------------------------------------------------
# 6. Mass function convention check
# ---------------------------------------------------------------------------

_header("MASS FUNCTION CONVENTION CHECK")
print("Recomputes f(m) = (M2 sin i)^3 / (M2+Mg)^2 from median solution.")
print(f"\n{'Target':<12} {'f(m) pred':>12} {'f(m) obs':>12} {'ratio':>8}")
print("-" * 46)

for name in ["V1472 Aql", "EG And", "BD Cam"]:
    cfg  = CONFIGS[name]
    r    = run_inference(cfg, GRID, n_samples=N_SENS)
    fm_pred = (r.m2_med * r.sin_i_med)**3 / (r.m2_med + r.mg_med)**2
    fm_obs  = cfg.orbit.fm
    print(f"{name:<12} {fm_pred:>12.4f} {fm_obs:>12.4f} {fm_pred/fm_obs:>8.2f}")

print("\n(ratio near 1 confirms convention: Mg = giant primary in SB1 f(m))")
print("\nDone.")
