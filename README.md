# ASTEX stratocumulus-to-cumulus transition in Breeze.jl

Large-eddy simulation (LES) of the **GASS/EUCLIPSE ASTEX First Lagrangian** stratocumulus-to-cumulus
transition, following [van der Dussen et al. (2013), *JAMES*](https://doi.org/10.1002/jame.20033),
built with [Breeze.jl](https://github.com/glwagner/Breeze.jl).

---

## Table of contents

1. [Scientific background](#scientific-background)
2. [What the code does](#what-the-code-does)
3. [Repository layout](#repository-layout)
4. [Requirements and setup](#requirements-and-setup)
5. [The case-data file](#the-case-data-file)
6. [How the code works](#how-the-code-works)
   - [`astex_case.jl` — case data](#astex_casejl--case-data)
   - [`astex.jl` — the simulation](#astexjl--the-simulation)
   - [`analyze_astex.jl` — diagnostics](#analyze_astexjl--diagnostics)
7. [Running the simulation](#running-the-simulation)
8. [Configuration: `dev` vs `production`](#configuration-dev-vs-production)
9. [Output files](#output-files)
10. [Validating against the paper](#validating-against-the-paper)
11. [Customizing the setup](#customizing-the-setup)
12. [Known simplifications](#known-simplifications)
13. [Troubleshooting](#troubleshooting)
14. [Reference](#reference)

---

## Scientific background

Subtropical marine boundary layers undergo a characteristic transition as air is advected
equatorward over progressively warmer water: an initially solid, well-mixed **stratocumulus** deck
deepens, decouples, and gives way to **shallow cumulus** convection that penetrates a thinning,
breaking stratocumulus layer. The ASTEX First Lagrangian (Atlantic Stratocumulus Transition
Experiment, June 1992) tracked an air mass through exactly this transition, and van der Dussen
et al. (2013) used it as the basis of a six-model GASS/EUCLIPSE LES intercomparison.

The transition is controlled by a competition between processes:

- **Cloud-top longwave radiative cooling** destabilizes the layer and sustains the turbulence that
  keeps it well mixed — this is why **interactive radiation is essential** and cannot be replaced by
  a prescribed cooling profile.
- A **warming sea surface** increases surface fluxes and deepens the layer.
- **Weakening large-scale subsidence** allows the inversion to rise.
- **Entrainment** of warm, dry free-tropospheric air at the inversion dries and warms the layer,
  eventually decoupling the cloud from its surface moisture source and triggering cumulus.

This repository reproduces that case with a cloud-resolving LES.

## What the code does

- **Anelastic LES** (Breeze's `AnelasticDynamics`) with **liquid-ice potential temperature**
  thermodynamics.
- **Warm-phase 1-moment microphysics** with prognostic rain (autoconversion + accretion +
  sedimentation), so drizzle — important late in the ASTEX transition — is represented.
- **Interactive RRTMGP all-sky radiation.** The cloud-droplet effective radius is *diagnosed* from
  the prescribed droplet number concentration `Nc = 100 cm⁻³` using
  `r_eff = 1.03 · r_vol`, `r_vol = (3 ρ qˡ / (4π ρ_w Nc))^{1/3}` — the relation specified in the
  EUCLIPSE setup. The ozone profile is read from the case file.
- **Time-varying large-scale forcings**, refreshed hourly from the case time series:
  - sea-surface temperature `SST(t)`,
  - large-scale divergence `D(t)` → subsidence `wˢ(z,t) = −D(t)·min(z, 1600 m)` (constant above
    1600 m),
  - geostrophic wind `u_g(t), v_g(t)`.
- A **top sponge layer** that damps vertical velocity and relaxes the horizontal wind toward the
  observed free-atmosphere wind `u_fa(t), v_fa(t)` (avoiding the spurious shear that whole-column
  geostrophic nudging would create — see the EUCLIPSE notes).
- **Bulk surface fluxes** of momentum, heat, and moisture computed from the SST with a
  wind/stability-dependent exchange coefficient.

## Repository layout

```
StratocumulusTransition/
├── README.md                     # this file
├── Project.toml                  # Julia environment (local Breeze + Oceananigans sources)
├── astex_case.jl                 # loads the case data; profile/time interpolators + constants
├── astex.jl                      # the simulation (dev/production switch via ASTEX_CONFIG)
├── analyze_astex.jl              # post-processing: profiles, time series, slice animation
├── run_astex_debug.sbatch        # GPU debug-queue smoke test (ASTEX_CONFIG=dev)
├── run_astex.sbatch              # Perlmutter GPU batch job for the production run
└── case_data/
    └── astex_input_v5.nc         # authoritative EUCLIPSE/GASS ASTEX case file
```

## Requirements and setup

- **Julia ≥ 1.11.9** (Breeze.jl's lower bound). On the development machine this is the juliaup
  channel `1.12.4`.
- A **local checkout of Breeze.jl and Oceananigans.jl**, referenced through the `[sources]` block of
  `Project.toml` (edit those paths for your machine). For NVIDIA GPU runs, a working CUDA stack.
- **A small Breeze.jl addition.** This case relies on a droplet-number-based effective-radius model,
  `DropletNumberConcentrationRadius`, added to Breeze alongside this project (it diagnoses the
  effective radius from `Nc` and the local cloud-liquid field, and threads `ρ, qˡ, qⁱ` through the
  all-sky radiation kernel — backward-compatible with the existing `ConstantRadiusParticles`). Make
  sure your Breeze checkout includes it.

Instantiate the environment:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

> The project intentionally has **no MPI dependency**, so `NCDatasets` resolves to the default
> (non-MPI) NetCDF build and precompiles cleanly.

## The case-data file

`case_data/astex_input_v5.nc` is the authoritative EUCLIPSE/GASS ASTEX case file
(downloaded from <http://www.euclipse.nl/wp3/ASTEX_Lagrangian/>). The simulation reads everything
from it — nothing physical is hard-coded — so the setup stays faithful to the intercomparison
protocol. Contents:

| Variable | Dimensions | Meaning |
|----------|------------|---------|
| `height` | `lev` | heights of the initial profiles (extends to ~49 km for radiation) |
| `thetal`, `qt`, `u`, `v` | `lev` | initial θ_l, total water, and wind profiles |
| `o3mmr` | `lev` | ozone **mass** mixing ratio (converted to volume mixing ratio for RRTMGP) |
| `tsec` | `time` | time of the hourly forcing series (41 points, 0–40 h) |
| `Tg` | `time` | sea-surface temperature |
| `div` | `time` | large-scale divergence (subsidence below 1600 m, constant above) |
| `ug`, `vg` | `time` | geostrophic wind |
| `ufa`, `vfa` | `time` | free-atmosphere wind (sponge-layer nudging target) |
| `zf`, `zh` | `zf`, `zh` | recommended LES full/half levels (cell centers/interfaces) |

The initial state: a well-mixed layer with `θ_l = 288 K`, `q_t = 10.2 g/kg` below the inversion at
`≈ 662 m`, jumping to `θ_l = 293.5 K`, `q_t = 9.1 g/kg` just above, with the free-tropospheric
profiles continuing aloft.

## How the code works

### `astex_case.jl` — case data

A small module (`ASTEXCase`) that reads the NetCDF file once and returns a `NamedTuple` of
ready-to-use closures:

- **profile interpolators** `θl(z), qt(z), u(z), v(z), O₃(z)` — linear in height, clamped at the
  ends. Ozone is converted from the file's mass mixing ratio to the **volume** mixing ratio RRTMGP
  expects (`× M_air / M_O₃`).
- **time interpolators** `SST(t), D(t), ug(t), vg(t), ufa(t), vfa(t)` — linear in seconds.
- `production_z_faces` — the cell interfaces of the recommended LES grid (from the file's half
  levels).
- `constants` — latitude/longitude, surface pressure, droplet concentration, inversion height,
  solar constant, duration, and the start date.

`stretched_faces([(top, Δz), …])` builds a coarse vertical grid for the development configuration.

### `astex.jl` — the simulation

Reads `ASTEX_CONFIG` (`dev` by default, or `production`) and assembles the model step by step:

1. **Grid** — `RectilinearGrid`, periodic in the horizontal, bounded in the vertical, with the
   case grid (production) or a coarse stretched grid (dev). Float32.
2. **Reference state & dynamics** — `ReferenceState` at `p_s = 1029 hPa`, `AnelasticDynamics`.
3. **Microphysics** — `OneMomentCloudMicrophysics` with saturation-adjustment cloud formation;
   bounds-preserving WENO advection for the moisture variables.
4. **Sea-surface temperature** — stored in a 2D `Field` (`SST`) shared by both the surface fluxes
   and the radiation model; updated hourly.
5. **Radiation** — `RadiativeTransferModel` with `AllSkyOptics`, the `Nc`-based effective radius,
   the ozone background atmosphere, ASTEX solar geometry, and a configurable update interval.
6. **Surface fluxes** — `BulkDrag`, `BulkSensibleHeatFlux`, `BulkVaporFlux` with a
   `PolynomialCoefficient` (wind/stability-dependent) driven by `SST`.
7. **Forcings** — `SubsidenceForcing`, `geostrophic_forcings`, and a `Relaxation` top sponge.
   The subsidence- and geostrophic-velocity `Field`s are mutated hourly; the sponge nudging targets
   read the clock time directly.
8. **Initial conditions** — profiles from the case file, with small random θ_l and q_t
   perturbations below the inversion to seed turbulence.
9. **Time-varying callback** (`update_forcings!`, hourly) — refreshes `SST`, the subsidence
   velocity, and the geostrophic wind from the case time series at the current model time.
10. **Output** — horizontally-averaged profiles, a column liquid-water-path field, and x–z / x–y
    slices.

### `analyze_astex.jl` — diagnostics

Reads the output and produces, for the chosen `ASTEX_CONFIG`:

- `*_profiles.png` — hourly mean profiles of θ_l, q_t, cloud liquid, and resolved `⟨w²⟩`.
- `*_timeseries.png` — mean liquid water path, cloud cover (fraction of columns with
  `LWP > 20 g/m²`), and inversion height (level of strongest θ_l gradient) vs time.
- `*_slices.mp4` — an animation of vertical velocity and cloud liquid in an x–z plane.

## Running the simulation

```bash
# 1. Instantiate (once)
julia --project -e 'using Pkg; Pkg.instantiate()'

# 2. Development run — coarse, ~2 h sim time, CPU-friendly (uses GPU if available)
ASTEX_CONFIG=dev julia --project astex.jl
ASTEX_CONFIG=dev julia --project analyze_astex.jl

# 3. Production run — full paper configuration, 40 h, on a GPU
sbatch run_astex.sbatch
ASTEX_CONFIG=production julia --project analyze_astex.jl
```

On a Slurm cluster (e.g. NERSC Perlmutter) prefer the fast-turnaround **debug** queue for the
smoke test rather than the login node:

```bash
sbatch run_astex_debug.sbatch     # GPU debug queue, 30-min limit, runs ASTEX_CONFIG=dev
```

Environment variables read by the scripts:

| Variable | Meaning |
|----------|---------|
| `ASTEX_CONFIG` | `dev` (default) or `production` |
| `ASTEX_STOP` | optional stop time in **hours**, overriding the config (e.g. `ASTEX_STOP=0.02` for a ~70 s build check) |

The batch scripts also set the juliaup Julia 1.12.4 binary and the scratch Julia depot.

## Configuration: `dev` vs `production`

| | `dev` | `production` |
|---|---|---|
| Horizontal | 32 × 32, 140 m | 128 × 128, 35 m |
| Domain | 4.48 × 4.48 km² | 4.48 × 4.48 km² |
| Vertical grid | coarse stretched (~58 levels) | EUCLIPSE grid, ≈ 5 m at inversion (~426 levels) |
| Duration | 2 h | 40 h |
| Radiation update | every 20 steps | every 5 steps |
| Architecture | GPU if available, else CPU | GPU |

The development configuration is for quickly checking that the model runs, stays stable, forms a
cloud layer, and produces cloud-top radiative cooling — not for quantitative comparison with the
paper.

## Output files

Written to the working directory, prefixed by the configuration (`astex_dev_*` / `astex_production_*`):

- `*_profiles.jld2` — 30-minute-averaged horizontal-mean profiles: `θl`, `qᵛ`, `qˡ` (total liquid),
  `qᶜˡ` (cloud liquid), `qʳ` (rain), `qᵗ`, `u`, `v`, `w²`, and mean `LWP`.
- `*_slices.jld2` — 5-minute x–z and x–y slices of `w` and cloud liquid, plus the 2D `LWP` field.

Load them with `FieldTimeSeries("astex_dev_profiles.jld2", "θl")`, etc.

## Validating against the paper

Per Breeze's validation guidance, work up from a short run:

1. **Initial-condition check** — the loader has been verified to reproduce the case file exactly
   (mixed-layer θ_l/q_t, the inversion jump, ozone, SST/divergence/wind series).
2. **Dev smoke run** — confirm no NaNs, reasonable `|w|`, a persistent cloud layer (`LWP > 0`), and
   that radiation cools the cloud top.
3. **Production run** — compare the **liquid-water-path, cloud-cover, and inversion-height time
   series** and the **profile evolution** to the transition figures in van der Dussen et al. (2013):
   gradual boundary-layer deepening, a negative buoyancy flux at the top of the subcloud layer, and
   a developing double-peaked vertical-velocity-variance profile.

## Customizing the setup

- **Resolution / duration / architecture** — edit the `:dev` / `:production` branch at the top of
  `astex.jl`.
- **Radiation cost** — increase `radiation_interval` (radiation is the most expensive component).
- **Microphysics** — swap `OneMomentCloudMicrophysics` for `SaturationAdjustment` (no precipitation)
  for a cheaper, non-drizzling variant.
- **Droplet concentration** — change `c.droplet_concentration` in `astex_case.jl` (feeds both the
  effective radius and, in principle, droplet sedimentation).
- **Domain translation** — the EUCLIPSE protocol suggests translating the domain by `(−2, −7) m/s`
  to reduce mean-wind advection error. It is left off here; enable it by offsetting the initial,
  geostrophic, and sponge-target winds by the translation velocity.

## Known simplifications

- **Cloud-droplet sedimentation** (with `σ_g = 1.2`) is not included in the 1-moment scheme
  (rain sedimentation is); this is a candidate enhancement.
- Ozone is converted from mass to volume mixing ratio for RRTMGP; water vapor for radiation comes
  from the model's prognostic moisture field.
- Free-atmosphere wind nudging is applied only inside the top sponge.

## Reference

> van der Dussen, J. J., Siebesma, A. P., et al. (2013). The GASS/EUCLIPSE model intercomparison of
> the stratocumulus transition as observed during ASTEX: LES results. *Journal of Advances in
> Modeling Earth Systems*, 5, 483–499. <https://doi.org/10.1002/jame.20033>

Case data: EUCLIPSE WP3 ASTEX Lagrangian intercomparison,
<http://www.euclipse.nl/wp3/ASTEX_Lagrangian/>.
