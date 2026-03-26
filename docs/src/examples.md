# Examples

The package ships with manuscript-oriented example scripts under `GPRADFWI.jl/examples/`. They double as the quickest way to understand the intended workflow.

## Core Validation Scripts

- `validate_forward.jl`: forward-model checks, B-scan generation, and field snapshots
- `validate_gradients.jl`: AD-versus-finite-difference gradient verification
- `run_hand_adjoint_baseline.jl`: reduced benchmark against a hand-coded adjoint
- `run_revision_validations.jl`: reduced full-physics AD-versus-FD comparator

## Inversion Scripts

- `run_fwi_large_domain.jl`: main multi-source dispersive inversion
- `run_fwi_nondispersive.jl`: non-dispersive comparator
- `run_fwi_nondispersive_eps_s.jl`: static-permittivity non-dispersive comparator
- `run_fwi_two_anomaly.jl`: two-target reconstruction study
- `run_fwi_noisy_multiseed.jl`: noise-robustness study
- `run_fwi_uncertainty.jl`: prior-misspecification stress tests
- `run_fwi_joint.jl`: joint permittivity-conductivity inversion
- `run_fwi_joint_mitigation.jl`: conductivity damping mitigation

## Typical Package Workflow

1. Build an `FDTDConfig` with `create_config`.
2. Generate the source time series with `create_source`.
3. Create material maps for `eps_inf`, `deps`, `tau`, and `sigma`.
4. Run `run_forward!` to simulate receiver data.
5. Define an inversion mask and choose `:eps_inf`, `:sigma`, or `:both`.
6. Use `fd_gradient` for debugging or `ad_gradient` / `run_fwi_multisource` for production runs.

## Reproducibility Notes

- The paper figures are generated from the example scripts together with the `../paper/data/` and `../paper/figs/` assets in the parent repository.
- Large examples are computationally expensive; start with the validation scripts before running the full inversion studies.
- The package tests remain the fastest automated check for basic correctness:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Building This Docs Site

```bash
julia --project=docs docs/make.jl
```
