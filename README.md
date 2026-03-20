# GPRADFWI.jl

Automatic-differentiation-enabled full-waveform inversion (FWI) for ground-penetrating radar (GPR) in dispersive media.

This package implements a 2D TM-mode finite-difference time-domain (FDTD) forward solver with single-pole Debye dispersion and convolutional perfectly matched layer (CPML) absorbing boundaries, coupled with compiler-level reverse-mode automatic differentiation via [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for gradient computation. No hand-derived adjoint equations are required.

**Reference:**
> J. W. Liu, "Automatic-differentiation-enabled full-waveform inversion for ground-penetrating radar in dispersive media," *Computers & Geosciences*, submitted, 2026.

## Features

- **Forward solver**: 2D TM-mode FDTD on a Yee grid with leapfrog time-stepping
- **Debye dispersion**: single-pole auxiliary differential equation (ADE) for frequency-dependent soil permittivity
- **CPML boundaries**: complex-frequency-shifted convolutional PML with polynomial grading
- **AD gradients**: Enzyme.jl reverse-mode differentiation through the full FDTD time-stepping loop, including Debye ADE and CPML updates
- **FD gradients**: central finite-difference reference implementation for validation
- **L-BFGS inversion**: bounded multi-source FWI with Tikhonov smoothness regularization, Armijo backtracking line search, and per-parameter bounds projection
- **Joint inversion**: simultaneous recovery of permittivity and conductivity with independent regularization weights and bounds

## Requirements

- Julia 1.10 or later
- Enzyme.jl (installed automatically via Project.toml)

Tested on Julia 1.12.5 with Enzyme 0.13.129 on macOS (Apple M3, arm64) and Linux (x86_64).

## Installation

Clone the repository and activate the package environment:

```bash
git clone https://github.com/[REPOSITORY_URL]/GPRADFWI.jl.git
cd GPRADFWI.jl
```

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick start

### Forward simulation

```julia
using GPRADFWI

# Define grid: 200 x 170 cells, 5 mm spacing, 500 MHz Ricker source
config = create_config(
    nx=200, ny=170, dx=0.005, fc=500e6, npml=15,
    src_ix=100, src_iy=25,
    rx_iy=25, rx_ix_list=collect(20:4:180),
)

# Material maps (nx x ny)
eps_inf = 4.0 * ones(200, 170)   # high-frequency relative permittivity
deps    = 4.0 * ones(200, 170)   # Debye relaxation strength
tau     = 0.3e-9 * ones(200, 170) # relaxation time [s]
sigma   = 0.005 * ones(200, 170) # conductivity [S/m]

# Run forward simulation
src_waveform = create_source(config)
rec_data = run_forward!(config, eps_inf, deps, tau, sigma, src_waveform)
# rec_data is (nt x n_receivers) matrix of recorded Ez
```

### AD gradient computation

```julia
using Enzyme

# Define parameter mask (cells to invert)
param_mask = falses(200, 170)
param_mask[75:125, 50:90] .= true
n_params = count(param_mask)

# Pack parameters into flat vector
x = [eps_inf[i,j] for j in 1:170 for i in 1:200 if param_mask[i,j]]

# Define objective: observed data from a "true" model
obs_data = ...  # from a forward run with the true model

function objective(x_vec)
    return forward_misfit(x_vec, config, obs_data, src_waveform,
                          eps_inf, deps, tau, sigma, param_mask, :eps_inf)
end

# Enzyme reverse-mode AD gradient
grad_ad = ad_gradient(objective, x)

# Finite-difference gradient (for validation)
grad_fd = fd_gradient(objective, x)
```

### Full-waveform inversion

```julia
# Multi-source FWI with Tikhonov regularization and bounds
result = run_fwi_multisource(
    configs,            # Vector of FDTDConfig (one per source position)
    obs_datas,          # Vector of observed data matrices
    src_waveforms,      # Vector of source waveforms
    eps_inf_init,       # Initial permittivity model
    deps_map, tau_map,  # Fixed Debye parameters
    sigma_init,         # Initial conductivity model
    param_mask;         # Inversion region
    max_iter=50,
    param_type=:eps_inf,  # :eps_inf, :sigma, or :both
    use_ad=true,
    lambda=1.0,           # Tikhonov weight for permittivity
    lower_bound=1.0,      # Physical bounds
    upper_bound=25.0,
)

# Result fields:
# result.eps_inf_est    - reconstructed permittivity map
# result.loss_history   - convergence history
# result.n_iter         - iterations completed
```

## Package structure

```
GPRADFWI.jl/
  src/
    GPRADFWI.jl       Module definition and exports
    types.jl           Data structures (FDTDConfig, DebyeMedium, FWIResult, ...)
    sources.jl         Ricker wavelet generation
    debye.jl           Debye ADE coefficient computation
    cpml.jl            CPML parameter initialization and grading profiles
    fdtd.jl            2D TM-mode FDTD forward solver
    inversion.jl       AD/FD gradient computation, L-BFGS optimizer, Tikhonov regularization
  test/
    runtests.jl        Test suite entry point
    test_constants.jl  Physical constant verification
    test_sources.jl    Ricker wavelet tests
    test_debye.jl      Debye coefficient tests
    test_cpml.jl       CPML parameter tests
    test_fdtd.jl       Forward solver tests (energy decay, reciprocity)
    test_inversion.jl  Gradient verification (AD vs FD)
  examples/
    validate_forward.jl              Forward solver validation (B-scan, field snapshots)
    validate_gradients.jl            AD gradient verification (non-dispersive + dispersive)
    run_fwi_large_domain.jl          Multi-source FWI on 200x170 domain
    run_fwi_nondispersive.jl         Dispersive vs non-dispersive comparison
    run_fwi_noisy_multiseed.jl       Noise robustness study (10 seeds x 3 SNR levels)
    run_fwi_joint.jl                 Joint permittivity-conductivity inversion
    run_fwi_joint_mitigation.jl      Joint inversion with conductivity norm damping
    run_fwi_uncertainty.jl           Debye mismatch and boundary-shift stress tests
    run_hand_adjoint_baseline.jl     Hand-coded adjoint baseline comparison
    run_revision_validations.jl      Full-physics AD-vs-FD comparator
    benchmark_timing.jl              Gradient computation timing benchmarks
    figure_generation.jl             Generate all paper figures
    figure_generation_joint.jl       Generate joint inversion figures
```

## Reproducing paper results

Each numerical experiment in the paper corresponds to an example script. To reproduce all results:

```bash
# Forward solver validation (Section III-A): B-scan, field snapshots, material maps
julia --project=. examples/validate_forward.jl

# AD gradient verification (Section III-B): non-dispersive + dispersive cases
julia --project=. examples/validate_gradients.jl

# Hand-adjoint baseline (Section III-C): AD vs hand-adjoint vs FD
julia --project=. examples/run_hand_adjoint_baseline.jl

# Full-physics AD-vs-FD comparator (Section III-D)
julia --project=. examples/run_revision_validations.jl

# Multi-source FWI (Section III-E): 200x170 domain, 5 sources, 50 iterations
julia --project=. examples/run_fwi_large_domain.jl

# Dispersive vs non-dispersive comparison (Section III-F)
julia --project=. examples/run_fwi_nondispersive.jl

# Uncertainty stress tests (Section III-G)
julia --project=. examples/run_fwi_uncertainty.jl

# Noise robustness (Section III-H): 10 seeds x {40, 30, 20} dB
julia --project=. examples/run_fwi_noisy_multiseed.jl

# Joint permittivity-conductivity inversion (Section III-I)
julia --project=. examples/run_fwi_joint.jl

# Joint inversion with sigma damping (Section III-I)
julia --project=. examples/run_fwi_joint_mitigation.jl
```

Note: the multi-source FWI (`run_fwi_large_domain.jl`) takes approximately 5.5 hours on a single CPU core. The noise robustness study runs 30 independent inversions. Enzyme's first AD compilation takes ~5 minutes; subsequent iterations are ~4--5 minutes each on the 200x170 domain.

## Running tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The test suite verifies physical constants, source waveforms, Debye coefficients, CPML parameters, forward solver stability and reciprocity, and AD gradient correctness against finite differences.

## Memory considerations

Enzyme reverse-mode AD stores the full forward trajectory for backpropagation. Approximate peak memory usage:

| Domain size  | Time steps | Peak memory |
|-------------|-----------|-------------|
| 60 x 50     | 300       | < 1 GB      |
| 200 x 170   | 1028      | ~12 GB      |
| 400 x 300   | 1714      | > 16 GB (OOM on 24 GB) |

For the 200x170 FWI domain, per-source gradient accumulation keeps peak memory at the single-source level (~12 GB). The 400x300 domain is used only for forward simulation (no AD).

## Method

The forward solver implements the standard Yee FDTD scheme for 2D TM-mode Maxwell's equations. Debye dispersion is incorporated via a semi-implicit auxiliary differential equation, yielding a combined E-field/polarization update that avoids separate displacement-field arrays. The CPML uses polynomial-graded conductivity and stretching profiles with recursive convolution updates.

Enzyme operates at the LLVM compiler level, transforming the compiled forward solver into its reverse-mode adjoint. This differentiates through the entire time-stepping loop---including Debye ADE updates, CPML boundary updates, and source injection---without hand-derived adjoint equations. Changing the forward physics (e.g., adding multi-pole Debye or Cole-Cole dispersion) requires only modifying the forward solver; the AD gradient follows automatically.

The L-BFGS optimizer uses a memory depth of 5, Armijo backtracking line search, safeguarded descent direction checks, and best-iterate tracking. Tikhonov smoothness regularization penalizes spatial gradients of the parameter field, with the regularization gradient computed analytically (discrete Laplacian).

## License

MIT License. See [LICENSE](LICENSE).
