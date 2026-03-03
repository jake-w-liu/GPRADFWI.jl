# debye.jl — Debye dispersive medium via Auxiliary Differential Equation (ADE)
#
# Single-pole Debye model:
#   ε(ω) = ε∞ + Δε / (1 + iωτ)
#
# Time-domain ADE:
#   P + τ ∂P/∂t = ε₀ Δε E
#
# Semi-implicit discretization (trapping rule for stability):
#   P^{n+1} = c1 * P^n + c2 * E^{n+1}
#   J_p^{n+1/2} = (P^{n+1} - P^n) / dt   (polarization current)
#
# Coefficients:
#   c1 = (2τ - dt) / (2τ + dt)
#   c2 = 2 ε₀ Δε dt / (2τ + dt)
#
# E-field update incorporating Debye:
#   D^{n+1} = D^n + dt * (curl H - σE^n)
#   E^{n+1} = (D^{n+1} - P^{n+1}) / (ε₀ ε∞)
#
# For AD-friendliness, we avoid the D-field entirely and write an
# equivalent direct E-field update with polarization correction.

"""
    DebyeCoeffs

Pre-computed coefficients for Debye ADE time stepping at each grid point.
"""
struct DebyeCoeffs
    c1::Matrix{Float64}     # polarization update: P^{n+1} = c1*P^n + c2*E^{n+1}
    c2::Matrix{Float64}
    # E-field update coefficients incorporating Debye + conductivity
    ca::Matrix{Float64}     # E^{n+1} multiplier for E^n
    cb::Matrix{Float64}     # E^{n+1} multiplier for curl_H * dt
    # Polarization feedback
    cp::Matrix{Float64}     # E^{n+1} correction from P^n
end

"""
    init_debye_coeffs(eps_inf, deps, tau, sigma, dt, nx, ny)

Compute Debye ADE coefficients for each grid cell.

The update scheme is:
1. Compute tentative D from curl H
2. Update polarization P via ADE
3. Recover E from D and P

Combining into a single E-update for AD efficiency:

  E^{n+1} = ca * E^n + cb * (∂Hy/∂x - ∂Hx/∂y) + cp * P^n

where ca, cb, cp absorb ε∞, Δε, τ, σ, and dt.
"""
function init_debye_coeffs(eps_inf::Matrix{Float64}, deps::Matrix{Float64},
                           tau::Matrix{Float64}, sigma::Matrix{Float64},
                           dt::Float64, nx::Int, ny::Int)
    @assert size(eps_inf) == (nx, ny) "eps_inf size mismatch"
    @assert size(deps) == (nx, ny) "deps size mismatch"
    @assert size(tau) == (nx, ny) "tau size mismatch"
    @assert size(sigma) == (nx, ny) "sigma size mismatch"
    @assert dt > 0 "dt must be positive"
    c1 = Matrix{Float64}(undef, nx, ny)
    c2 = Matrix{Float64}(undef, nx, ny)
    ca = Matrix{Float64}(undef, nx, ny)
    cb = Matrix{Float64}(undef, nx, ny)
    cp = Matrix{Float64}(undef, nx, ny)

    for j in 1:ny, i in 1:nx
        ei  = eps_inf[i, j]
        de  = deps[i, j]
        ta  = tau[i, j]
        sg  = sigma[i, j]

        if ta > 0.0 && de != 0.0
            # Debye dispersive cell
            denom_p = 2.0 * ta + dt
            c1[i, j] = (2.0 * ta - dt) / denom_p
            c2[i, j] = 2.0 * eps0 * de * dt / denom_p

            # Combined E-field update:
            # From Maxwell: ε₀ε∞ * (E^{n+1} - E^n) + (P^{n+1} - P^n) + σ dt E^{n+1/2} = dt * curl H
            # Using semi-implicit for σ: σ dt E^{n+1/2} ≈ σ dt/2 (E^n + E^{n+1})
            # And P^{n+1} = c1 P^n + c2 E^{n+1}
            #
            # Gather E^{n+1} terms:
            # (ε₀ε∞ + c2 + σdt/2) E^{n+1} = (ε₀ε∞ - σdt/2) E^n
            #                                  + dt curl_H + (1 - c1) P^n
            denom = eps0 * ei + c2[i, j] + sg * dt / 2.0
            ca[i, j] = (eps0 * ei - sg * dt / 2.0) / denom
            cb[i, j] = dt / denom
            cp[i, j] = (1.0 - c1[i, j]) / denom
        else
            # Non-dispersive cell (simple lossy dielectric)
            c1[i, j] = 0.0
            c2[i, j] = 0.0
            denom = eps0 * ei + sg * dt / 2.0
            ca[i, j] = (eps0 * ei - sg * dt / 2.0) / denom
            cb[i, j] = dt / denom
            cp[i, j] = 0.0
        end
    end

    return DebyeCoeffs(c1, c2, ca, cb, cp)
end
