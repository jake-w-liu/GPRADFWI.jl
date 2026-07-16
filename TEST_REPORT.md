# GPRADFWI.jl Test Report

## Result

- Tests passed: **258**.
- Tests failed: **0**.
- Deterministic forward plus AD/FD smoke: passed.
- Smoke directional relative error: `7.664328e-11`.

## Tolerance Justification

- The smoke experiment accepts an AD/FD directional relative error no larger than `1e-3`. The observed `7.664328e-11` is well inside that bound and resolves the same scalar directional derivative through reverse AD and central finite differences.
- The returned-model consistency regression recomputes the objective from the returned material model and compares it with the last stored loss using `rtol=1e-12` and `atol=1e-12`. This tolerance permits only Float64 evaluation-order noise, not selection of a different optimizer iterate.
- CPML compact-index expectations use exact integer equality because the required map is discrete: the first active right/top H half-cell is index 1 and the complete strip is `1:npml`.
- Source-coupling expectations use directly constructed coefficients and independently evaluate `old_P + c2*delta_E`; the chosen values are exactly representable in the regression case.
- Finite-difference and hand-adjoint comparisons use independently computed derivatives. Their numerical tolerances account for finite-difference truncation and floating-point accumulation; they do not permit sign, scale, indexing, or omitted-state-update errors.
- Multipole reduction compares the one-pole multipole and single-pole execution paths under the test's numerical tolerance. The expected equivalence follows from the discrete model reduction, not from copying one output into the other expectation.

## Anti-False-Test Checks

- The CPML regression compares the implementation with the independently known one-based compact sequence. Removing the corrected `+1` changes the sequence to start at zero and fails the test.
- The source regression computes the polarization expectation separately from the source helper. Omitting `c2*delta_E` changes the expected source-cell state and fails the test.
- The instantaneous Debye regression exercises `tau=0` directly, so replacing the explicit limit with a divide-by-zero path or a small positive proxy changes the result.
- Invalid parameter symbols and both short and long packed vectors are rejected. A check that merely prevents out-of-bounds access cannot satisfy all three expectations.
- The optimizer regression recomputes the scalar objective from `eps_inf_est` and compares it with the final history entry. Returning a hidden retained-best model while preserving final-iterate histories is mutation-sensitive and fails this comparison when the iterates differ.
- Callback tests check invocation count, iteration numbers, final loss agreement, and parameter bounds. A callback fired on trial line-search points or exposing a stale model does not satisfy these expectations.
- Central finite differences and the matched hand adjoint are independent of Enzyme's reverse transformation. Agreement therefore does not reduce to comparing an Enzyme result with itself.
- The multipole one-pole comparison executes distinct single-pole and multipole update paths. It detects coefficient, state-accumulation, source-coupling, and receiver-sampling differences between the implementations.

These checks document mutation sensitivity and independent expectations present in the passing suite. No separate exhaustive mutation-testing campaign is claimed.

## Covered Corrections and Extensions

| Area | Regression target |
|------|-------------------|
| CPML | Right/top H half-cell compact indexing |
| Debye source | Coupled electric-field and polarization source increment |
| Debye limit | Instantaneous `tau=0` response |
| Multipole | Independent poles and one-pole reduction |
| Validation | Dimensions, indices, symbols, snapshot uniqueness, exact vector layout |
| Inversion | Sigma diagnostics, final-iterate return, accepted-iterate callback |
| Gradient | Enzyme, central finite differences, matched hand adjoint |
