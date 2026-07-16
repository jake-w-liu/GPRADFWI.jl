# GPRADFWI.jl Code Notes

## Development Ledger

### 2026-07-17: solver corrections

- Corrected the one-based compact index for the first active right and top CPML H half-cells. The active compact span now maps to `1:npml` rather than beginning at zero.
- Centralized source injection so the electric-field increment is accompanied by the constitutively required `c2*delta_E` polarization increment in forward and snapshot execution.
- Added preflight checks for material and observation dimensions, source and receiver indices, waveform length, CPML width, snapshot uniqueness, parameter symbols, masks, and exact packed-vector lengths.
- Added the explicit instantaneous `tau=0` Debye limit rather than approximating it with a small positive relaxation time.

### 2026-07-17: material and gradient extensions

- Added the multipole implementation in `src/multipole.jl` with public APIs `MultiPoleDebyeMedium`, `init_multipole_coeffs`, `discrete_debye_susceptibility`, `run_forward_multipole!`, and `multipole_forward_misfit_eps`.
- Added multipole coverage in `test/test_multipole.jl`, including reduction of a one-pole multipole medium to the single-pole forward behavior.
- Added the matched discrete hand adjoint in `src/hand_adjoint.jl`. The revision experiments use `run_forward_hand_tape`, `hand_adjoint_gradient`, and `hand_adjoint_store_all_bytes`.
- Added hand-adjoint coverage in `test/test_hand_adjoint.jl` as an independent comparator for Enzyme and finite differences.

### 2026-07-17: inversion consistency

- Both inversion drivers now return the final adopted iterate represented by the last loss-history entry; no retained-best model is substituted after history generation.
- Conductivity-only smoothness and damping use the conductivity weights and diagnostic channel.
- Both drivers accept an optional callback. It receives the initial state and each adopted iterate with copied packed parameters, total/data/regularization losses, gradient norm, step size, and backtrack count.

### 2026-07-17: verification state

- The package suite completed with 258 passed tests and no failures.
- The deterministic forward plus AD/FD smoke experiment passed with directional relative error `7.664328e-11`.
- Tests use exact analytical expectations where available, central finite differences and a matched hand adjoint as independent gradient comparators, and regression expectations that change under the corrected CPML, source-coupling, validation, and optimizer mutations.

## Current Design Contracts

- Forward and snapshot solvers share the same Debye-consistent source operation.
- Single-pole and multipole paths agree in the one-pole limit.
- `ad_gradient` is the Enzyme path; finite differences remain an explicit independent comparator rather than a hidden fallback.
- Callback parameter arrays are copies and cannot be changed by later optimizer updates.
- Allocated bytes, post-GC Julia live heap, process RSS, and peak footprint are distinct memory measurements and must not be substituted for one another.

