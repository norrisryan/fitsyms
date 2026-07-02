# FITSYMS

**FITting SYMbiotic Systems** — joint inference of giant star masses,
companion masses, and Roche-lobe filling factors from interferometric angular
diameters, spectroscopic mass functions, and MIST stellar isochrone grids.

FITSYMS implements the methodology of Gaudin et al. (2026, ApJ, submitted).
If you use this code, please cite that paper.

---

## What it does

For each target, FITSYMS combines:

- **Interferometric angular diameter** θ_LD (from CHARA/MIRC-X or similar)
- **Distance** (Gaia EDR3 Bailer-Jones+2021 or eclipse/bolometric)
- **Effective temperature** T_eff (from spectral fitting, e.g. Starfish)
- **[Fe/H]** prior (literature spectroscopy or Galactic MDF)
- **Spectroscopic mass function** f(m) with period and eccentricity
- **MIST v1.2 giant-branch isochrone grid** (loaded directly from `_full.iso` files)

The inference uses importance sampling: proposal draws in (θ, d, T_eff, [Fe/H])
are mapped to (logT, logL, [Fe/H]) and matched to the nearest MIST giant-branch
nodes to assign M_g.  The mass function f(m) = (M₂ sin i)³/(M₁ + M₂)² provides
the importance weights, giving a joint posterior over (M_g, M₂, R, f = R/R_L).

The key methodological choice is using `star_mass` (current post-wind-loss mass)
rather than `initial_mass` from the MIST grid.  `star_mass` is only available in
the `_full.iso` files — not via the `isochrones` library — which is why FITSYMS
reads MIST files directly with pandas.

---

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `pandas`, `h5py`, `pyyaml`, `matplotlib`
- MIST v1.2 `_full.iso` files (external; see below)
- Optional: `corner` (for plots in `paper/`)

---

## Installing MIST isochrone files

FITSYMS does **not** bundle MIST isochrone files.  Download them from the MIST
website:

**Download URL:** https://waps.cfa.harvard.edu/MIST/model_grids.html

Under **Isochrones**, select:

| Setting | Value |
|---------|-------|
| Version | v1.2 |
| Rotation | v/vcrit = 0.4 |
| Helium | Solar-scaled |
| Type | **Full isochrone grids** (`_full.iso`) |
| [Fe/H] range | −4.0 to +0.5 (or at minimum −1.0 to +0.5) |

Download the full-isochrone tar archive and extract.  The directory should
contain files named like:

```
MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.4_full.iso
MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_full.iso
...
```

Place (or symlink) the directory at:

```
~/.isochrones/mist/MIST_v1.2_vvcrit0.4_full_isos/
```

or set a custom path in your config YAML:

```yaml
mist_dir: "/data/mist/MIST_v1.2_vvcrit0.4_full_isos"
```

**Why `_full.iso` and not the standard isochrones?**
The standard MIST isochrone files and the `isochrones` Python library expose
only `initial_mass`.  The `_full.iso` files additionally carry `star_mass` —
the current post-wind-loss mass — which is the dynamically correct quantity
for the spectroscopic mass function.  Using `initial_mass` overstates M_g by
∼0.1–0.3 M☉ for RGB/AGB stars (see Gaudin et al. §4.2).

---

## Installation

```bash
# From GitHub (recommended)
pip install git+https://github.com/YOUR_USERNAME/fitsyms.git

# From a local clone
git clone https://github.com/YOUR_USERNAME/fitsyms.git
cd fitsyms
pip install -e ".[dev]"
```

---

## Quickstart

### 1. Write a config file

Copy and edit one of the provided examples in `config/`:

```bash
cp config/egand.yaml config/my_target.yaml
```

Key sections:

```yaml
name: "My Giant"
mist_dir: "~/.isochrones/mist/MIST_v1.2_vvcrit0.4_full_isos"
gardner_factor: 1.0054     # set to 1.0 if correction already applied

angular_diameter:
  theta_mas: 2.44          # raw LDD angular diameter [mas]
  err_lo: 0.05
  err_hi: 0.05

distance:
  dist_pc: 247.62
  err_lo: 4.0
  err_hi: 5.0
  source: "Gaia EDR3 BJ+21"

spectroscopy:
  teff: 3624
  teff_err: 117
  av: 0.607
  av_err: 0.090
  feh_starfish: 0.015
  feh_starfish_err: 0.259

feh_prior:
  kind: truncated_gaussian
  mu: +0.02
  sigma: 0.20
  feh_lo: -1.0
  feh_hi: +0.5
  cite: "Hayden et al. (2015)"

companion_prior:
  kind: flat
  lo: 0.30
  hi: 3.00

incl_prior:
  kind: isotropic_truncated
  incl_min: 50.0
  incl_max: 90.0

orbit:
  has_orbit: true
  fm: 0.045
  fm_err: 0.003
  period_d: 198.716
  period_err_d: 0.038
  ecc: 0.048
  ecc_err: 0.016

inference:
  n_samples: 1000000       # use 10000000 for published results
  k_neighbors: 16
  rng_seed: 20260501
```

### 2. Run from the command line

```bash
# Single target
fitsyms run config/egand.yaml --output results/

# Multiple targets
fitsyms run config/egand.yaml config/bdcam.yaml config/v1472aql.yaml \
    --output results/

# Quick test with fewer samples
fitsyms run config/egand.yaml --n-samples 100000 --output results/

# Print summary table from saved results
fitsyms summary results/
```

### 3. Run from Python

```python
from fitsyms import MISTGrid, TargetConfig, run_inference

# Load grid once (slow; reuse across targets)
grid = MISTGrid("~/.isochrones/mist/MIST_v1.2_vvcrit0.4_full_isos")

# Load config and run
config = TargetConfig.from_yaml("config/egand.yaml")
result = run_inference(config, grid)

# Print summary
print(result.summary_line())
# EG And       | Mg=1.05+0.18-0.14  M2=0.44+0.09-0.07  f=0.63+0.07-0.07  ESS=124832

# Save to HDF5 + JSON
result.save("results/egand.h5")

# Load later (posteriors optional)
from fitsyms.results import InferenceResult
r = InferenceResult.load("results/egand.h5")
import numpy as np
print(np.median(r.mg_post))
```

---

## Output files

Each run produces two files:

| File | Contents |
|------|----------|
| `<name>.h5` | HDF5: `/summary` (scalars) + `/posteriors` (full arrays, gzip-compressed float32) |
| `<name>.json` | JSON: scalar summary only (medians, percentiles, ESS, diagnostics) |

HDF5 posterior arrays:

| Dataset | Description |
|---------|-------------|
| `mg` | Giant mass M_g [M☉] |
| `m2` | Companion mass M₂ [M☉] |
| `ff` | Filling factor f = R/R_L (time-averaged) |
| `ff_peri` | Filling factor at periastron (conservative upper bound) |
| `R` | Giant radius [R☉] |
| `sini` | sin(i) |
| `dist` | Distance [pc] |
| `feh` | [Fe/H] |
| `teff` | T_eff [K] |

---

## Config reference

### [Fe/H] prior kinds

| `kind` | Parameters | When to use |
|--------|-----------|-------------|
| `truncated_gaussian` | `mu`, `sigma`, `feh_lo`, `feh_hi` | Literature spectroscopy or Galactic MDF prior (recommended) |
| `uniform` | `feh_lo`, `feh_hi` | No constraint; Starfish posterior is flat across emulator support |
| `gaussian` | `mu`, `sigma` | Sensitivity tests only |

### Inclination prior kinds

| `kind` | Parameters | When to use |
|--------|-----------|-------------|
| `isotropic_truncated` | `incl_min`, `incl_max` | Eclipsing systems with inclination lower bound |
| `gaussian_deg` | `incl_deg`, `incl_err_deg` | Astrometric orbit (e.g. Hipparcos) |
| `uniform_deg` | `incl_min`, `incl_max` | Sensitivity tests only |
| `isotropic` | — | No constraint |

### Companion mass prior kinds

| `kind` | Parameters | When to use |
|--------|-----------|-------------|
| `flat` | `lo`, `hi` | Default (broad prior on WD or MS companion) |
| `gaussian` | `lo`, `hi`, `mu`, `sigma` | WD with photospheric mass from UV fitting |

---

## Running the Gaudin et al. sensitivity analysis

The paper's sensitivity tables (§5 and referee response appendix) are
reproduced by:

```bash
python paper/run_sensitivity.py | tee paper/sensitivity_output.txt
```

This uses N=1,000,000 samples per run (not N_MAIN=10M) for speed.
The main-analysis results at N_MAIN are produced by:

```bash
fitsyms run config/egand.yaml config/bdcam.yaml config/v1472aql.yaml \
    config/sulyn.yaml --output results/
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests cover all physics functions, samplers, config loading/validation,
and HDF5 save/load round-trips.  They do not require MIST files.

---

## Citation

If you use FITSYMS, please cite:

```bibtex
@article{gaudin2026,
  author  = {Gaudin, A. and Norris, R. P. and ...},
  title   = {Roche-lobe filling factors and masses of symbiotic giants from
             optical interferometry and MIST isochrones},
  journal = {The Astrophysical Journal},
  year    = {2026},
  note    = {submitted}
}
```

and the MIST isochrone papers:

```bibtex
@article{choi2016,
  author  = {Choi, J. and Dotter, A. and Conroy, C. and Cantiello, M.
             and Paxton, B. and Johnson, B. D.},
  title   = {{MESA} Isochrones and Stellar Tracks ({MIST}). {I}.
             Solar-scaled Models},
  journal = {ApJ},
  year    = {2016},
  volume  = {823},
  pages   = {102}
}

@article{dotter2016,
  author  = {Dotter, A.},
  title   = {{MESA} Isochrones and Stellar Tracks ({MIST}) 0: Methods for
             the Construction of Stellar Isochrones},
  journal = {ApJS},
  year    = {2016},
  volume  = {222},
  pages   = {8}
}
```

---

## License

MIT
