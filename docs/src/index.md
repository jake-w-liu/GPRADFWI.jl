# GPRADFWI.jl

`GPRADFWI.jl` is a Julia package for two-dimensional TM-mode ground-penetrating radar full-waveform inversion in dispersive media. The package combines:

- a Yee-grid FDTD forward solver,
- single-pole Debye dispersion through an auxiliary differential equation,
- CPML absorbing boundaries, and
- gradient computation through Enzyme.jl reverse-mode automatic differentiation.

This documentation is the package-level reference for the Julia code. The manuscript-facing workflow and paper assets live in the surrounding research repository, but this docs site focuses on the reusable package API and example entry points.

## Package Structure

- `src/types.jl`: simulation and inversion data structures
- `src/sources.jl`: source-waveform generation
- `src/debye.jl`: Debye ADE coefficients
- `src/cpml.jl`: CPML setup
- `src/fdtd.jl`: forward simulation and differentiable misfit
- `src/inversion.jl`: finite-difference gradients, AD gradients, and FWI drivers

## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Minimal Workflow

```julia
using GPRADFWI

config = create_config(
    nx = 200,
    ny = 170,
    dx = 0.005,
    fc = 500e6,
    npml = 10,
    src_ix = 100,
    src_iy = 25,
    rx_iy = 25,
    rx_ix_list = collect(20:4:180),
)

src = create_source(config)

eps_inf = 4.0 .* ones(config.nx, config.ny)
deps = 4.0 .* ones(config.nx, config.ny)
tau = 0.3e-9 .* ones(config.nx, config.ny)
sigma = 0.005 .* ones(config.nx, config.ny)

rec_data = run_forward!(config, eps_inf, deps, tau, sigma, src)
```

## Main Public Entry Points

- `create_config`: build a simulation configuration from grid, source, and receiver settings
- `create_source`: generate the source time series
- `run_forward!`: run the FDTD solver and record receiver traces
- `run_forward_snapshots`: run the forward solve while saving selected field snapshots
- `fd_gradient`: finite-difference reference gradient
- `ad_gradient`: Enzyme-based reverse-mode gradient for a scalar objective
- `run_fwi`: minimal single-source inversion helper
- `run_fwi_multisource`: bounded and regularized multi-source inversion driver used by the paper scripts

## Related Pages

- [Examples](examples.md): experiment scripts and reproducibility workflow
- [API](api.md): exported package types, functions, and constants
