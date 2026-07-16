# multipole.jl - Multi-pole Debye ADE extension for the 2D TM FDTD solver
#
# This file is intended to be included inside the GPRADFWI module. Pole maps
# use (nx, ny, npoles) storage so the time-stepping loops have fixed, dense
# indexing and allocate no temporary arrays.

"""
    MultiPoleDebyeMedium(deps, tau)

Validated multi-pole Debye material maps. `deps[i,j,p]` is the relative
permittivity strength and `tau[i,j,p]` is the relaxation time in seconds for
pole `p` at cell `(i,j)`. Active poles require `deps >= 0` and `tau > 0`;
zero-strength poles may use any nonnegative `tau`.
"""
struct MultiPoleDebyeMedium
    deps::Array{Float64,3}
    tau::Array{Float64,3}

    function MultiPoleDebyeMedium(deps::Array{Float64,3}, tau::Array{Float64,3})
        size(deps) == size(tau) || throw(DimensionMismatch("deps and tau must have identical shapes"))
        size(deps, 3) > 0 || throw(ArgumentError("at least one Debye pole is required"))

        for k in eachindex(deps, tau)
            de = deps[k]
            ta = tau[k]
            isfinite(de) || throw(ArgumentError("deps must contain only finite values"))
            isfinite(ta) || throw(ArgumentError("tau must contain only finite values"))
            de >= 0.0 || throw(ArgumentError("Debye strengths must be nonnegative"))
            ta >= 0.0 || throw(ArgumentError("relaxation times must be nonnegative"))
            de == 0.0 || ta > 0.0 || throw(ArgumentError("every active pole must have tau > 0"))
        end

        return new(deps, tau)
    end
end

function MultiPoleDebyeMedium(deps::AbstractArray{<:Real,3},
                              tau::AbstractArray{<:Real,3})
    return MultiPoleDebyeMedium(Array{Float64,3}(deps), Array{Float64,3}(tau))
end

"""
    MultiPoleDebyeCoeffs

Precomputed coefficients for
`P_p^(n+1) = c1_p P_p^n + c2_p E^(n+1)` and the combined electric-field
update. `cp_p P_p^n` is the feedback contribution from pole `p`.
"""
struct MultiPoleDebyeCoeffs
    c1::Array{Float64,3}
    c2::Array{Float64,3}
    ca::Matrix{Float64}
    cb::Matrix{Float64}
    cp::Array{Float64,3}
end

"""
    init_multipole_coeffs(eps_inf, medium, sigma, dt, nx, ny)

Precompute the semi-implicit multi-pole Debye coefficients. The resulting
update is the direct multi-pole generalization of `init_debye_coeffs`:

```
(eps0*eps_inf + sum(c2_p) + sigma*dt/2) Enew =
    (eps0*eps_inf - sigma*dt/2) Eold + dt*curl(H) +
    sum((1-c1_p) Pold_p).
```
"""
function init_multipole_coeffs(eps_inf::Matrix{Float64},
                               medium::MultiPoleDebyeMedium,
                               sigma::Matrix{Float64},
                               dt::Float64,
                               nx::Int,
                               ny::Int)
    @assert size(eps_inf) == (nx, ny) "eps_inf size mismatch"
    @assert size(sigma) == (nx, ny) "sigma size mismatch"
    @assert size(medium.deps, 1) == nx && size(medium.deps, 2) == ny "pole-map size mismatch"
    @assert dt > 0.0 "dt must be positive"

    npoles = size(medium.deps, 3)
    c1 = Array{Float64,3}(undef, nx, ny, npoles)
    c2 = Array{Float64,3}(undef, nx, ny, npoles)
    cp = Array{Float64,3}(undef, nx, ny, npoles)
    c2_sum = zeros(nx, ny)
    ca = Matrix{Float64}(undef, nx, ny)
    cb = Matrix{Float64}(undef, nx, ny)

    for p in 1:npoles, j in 1:ny, i in 1:nx
        de = medium.deps[i, j, p]
        ta = medium.tau[i, j, p]
        if de == 0.0
            c1[i, j, p] = 0.0
            c2[i, j, p] = 0.0
        else
            denom_p = 2.0 * ta + dt
            c1[i, j, p] = (2.0 * ta - dt) / denom_p
            c2[i, j, p] = 2.0 * eps0 * de * dt / denom_p
            c2_sum[i, j] += c2[i, j, p]
        end
    end

    for j in 1:ny, i in 1:nx
        denom = eps0 * eps_inf[i, j] + c2_sum[i, j] + sigma[i, j] * dt / 2.0
        ca[i, j] = (eps0 * eps_inf[i, j] - sigma[i, j] * dt / 2.0) / denom
        cb[i, j] = dt / denom
        for p in 1:npoles
            cp[i, j, p] = medium.deps[i, j, p] == 0.0 ? 0.0 :
                          (1.0 - c1[i, j, p]) / denom
        end
    end

    return MultiPoleDebyeCoeffs(c1, c2, ca, cb, cp)
end

"""
    discrete_debye_susceptibility(deps, tau, omega, dt)

Return the complex relative susceptibility of the implemented ADE recurrence
at angular frequency `omega` under the repository's `exp(+im*omega*t)`
convention. For `z = exp(im*omega*dt)`, each pole contributes
`(c2/eps0) * z/(z-c1)`.
"""
function discrete_debye_susceptibility(deps::AbstractVector{<:Real},
                                       tau::AbstractVector{<:Real},
                                       omega::Real,
                                       dt::Real)
    length(deps) == length(tau) || throw(DimensionMismatch("deps and tau lengths differ"))
    dt > 0.0 || throw(ArgumentError("dt must be positive"))
    omega >= 0.0 || throw(ArgumentError("omega must be nonnegative"))

    z = cis(Float64(omega * dt))
    chi = 0.0 + 0.0im
    for p in eachindex(deps, tau)
        de = Float64(deps[p])
        ta = Float64(tau[p])
        de >= 0.0 || throw(ArgumentError("Debye strengths must be nonnegative"))
        if de != 0.0
            ta > 0.0 || throw(ArgumentError("every active pole must have tau > 0"))
            denom_p = 2.0 * ta + Float64(dt)
            c1 = (2.0 * ta - Float64(dt)) / denom_p
            c2_over_eps0 = 2.0 * de * Float64(dt) / denom_p
            chi += c2_over_eps0 * z / (z - c1)
        end
    end
    return chi
end

function _update_E_multipole!(Ez, Hx, Hy, Pz, dc::MultiPoleDebyeCoeffs,
                              cpml, dx, dy, nx, ny, npoles)
    for j in 2:ny-1, i in 2:nx-1
        dHy = Hy[i, j] - Hy[i-1, j]
        dHx = Hx[i, j] - Hx[i, j-1]
        curl_H = dHy / (cpml.kappa_ex[i] * dx) - dHx / (cpml.kappa_ey[j] * dy)
        curl_H_cpml = 0.0

        if i <= size(cpml.psi_ezx_x1, 1)
            cpml.psi_ezx_x1[i, j] = cpml.be_x[i] * cpml.psi_ezx_x1[i, j] +
                                      cpml.ce_x[i] * dHy
            curl_H_cpml += cpml.psi_ezx_x1[i, j] / dx
        end
        ii = i - (nx - size(cpml.psi_ezx_x2, 1))
        if ii >= 1 && ii <= size(cpml.psi_ezx_x2, 1)
            cpml.psi_ezx_x2[ii, j] = cpml.be_x[i] * cpml.psi_ezx_x2[ii, j] +
                                       cpml.ce_x[i] * dHy
            curl_H_cpml += cpml.psi_ezx_x2[ii, j] / dx
        end

        if j <= size(cpml.psi_ezy_y1, 2)
            cpml.psi_ezy_y1[i, j] = cpml.be_y[j] * cpml.psi_ezy_y1[i, j] +
                                      cpml.ce_y[j] * dHx
            curl_H_cpml -= cpml.psi_ezy_y1[i, j] / dy
        end
        jj = j - (ny - size(cpml.psi_ezy_y2, 2))
        if jj >= 1 && jj <= size(cpml.psi_ezy_y2, 2)
            cpml.psi_ezy_y2[i, jj] = cpml.be_y[j] * cpml.psi_ezy_y2[i, jj] +
                                       cpml.ce_y[j] * dHx
            curl_H_cpml -= cpml.psi_ezy_y2[i, jj] / dy
        end

        feedback = 0.0
        for p in 1:npoles
            feedback += dc.cp[i, j, p] * Pz[i, j, p]
        end

        Ez[i, j] = dc.ca[i, j] * Ez[i, j] +
                   dc.cb[i, j] * (curl_H + curl_H_cpml) + feedback

        for p in 1:npoles
            Pz[i, j, p] = dc.c1[i, j, p] * Pz[i, j, p] +
                          dc.c2[i, j, p] * Ez[i, j]
        end
    end
    return nothing
end

"""
    run_forward_multipole!(config, eps_inf, medium, sigma, src_waveform)

Run the production 2D TM Yee-grid solver with CPML and any positive number of
Debye relaxation poles. Soft-source injection updates both `Ez` and every
polarization state, preserving the constitutive relation at the source cell.
"""
function run_forward_multipole!(config::FDTDConfig,
                                eps_inf::Matrix{Float64},
                                medium::MultiPoleDebyeMedium,
                                sigma::Matrix{Float64},
                                src_waveform::Vector{Float64})
    nx = config.nx
    ny = config.ny
    nt = config.nt
    dx = config.dx
    dy = config.dy
    npoles = size(medium.deps, 3)

    @assert size(eps_inf) == (nx, ny) "eps_inf must be $nx x $ny"
    @assert size(sigma) == (nx, ny) "sigma must be $nx x $ny"
    @assert size(medium.deps, 1) == nx && size(medium.deps, 2) == ny "pole-map size mismatch"
    @assert length(src_waveform) == nt "source length must be $nt"

    Ez = zeros(nx, ny)
    Hx = zeros(nx, ny)
    Hy = zeros(nx, ny)
    Pz = zeros(nx, ny, npoles)
    dc = init_multipole_coeffs(eps_inf, medium, sigma, config.dt, nx, ny)
    cpml = init_cpml(config)
    rec_data = zeros(nt, length(config.rx_ix))

    ch_dt_dx = config.dt / (mu0 * dx)
    ch_dt_dy = config.dt / (mu0 * dy)
    si = config.source.ix
    sj = config.source.iy

    for n in 1:nt
        _update_H!(Hx, Hy, Ez, ch_dt_dx, ch_dt_dy, cpml, nx, ny)
        _update_E_multipole!(Ez, Hx, Hy, Pz, dc, cpml, dx, dy, nx, ny, npoles)

        delta_E = dc.cb[si, sj] * src_waveform[n] / (dx * dy)
        Ez[si, sj] += delta_E
        for p in 1:npoles
            Pz[si, sj, p] += dc.c2[si, sj, p] * delta_E
        end

        for r in eachindex(config.rx_ix)
            rec_data[n, r] = Ez[config.rx_ix[r], config.rx_iy[r]]
        end
    end

    return rec_data
end

"""
    multipole_forward_misfit_eps(params_flat, config, obs_data, src_waveform,
                                 eps_inf_bg, medium, sigma, param_mask)

Differentiable L2 objective for `eps_inf` inversion through the complete
multi-pole Debye plus CPML solver. The concrete signature is suitable for a
direct `Enzyme.autodiff` call with every argument except `params_flat` marked
constant.
"""
function multipole_forward_misfit_eps(params_flat::Vector{Float64},
                                      config::FDTDConfig,
                                      obs_data::Matrix{Float64},
                                      src_waveform::Vector{Float64},
                                      eps_inf_bg::Matrix{Float64},
                                      medium::MultiPoleDebyeMedium,
                                      sigma::Matrix{Float64},
                                      param_mask::BitMatrix)
    eps_inf = copy(eps_inf_bg)
    idx = 1
    for j in 1:config.ny, i in 1:config.nx
        if param_mask[i, j]
            eps_inf[i, j] = params_flat[idx]
            idx += 1
        end
    end

    syn_data = run_forward_multipole!(config, eps_inf, medium, sigma, src_waveform)
    misfit = 0.0
    for j in axes(obs_data, 2), i in axes(obs_data, 1)
        residual = syn_data[i, j] - obs_data[i, j]
        misfit += residual * residual
    end
    return 0.5 * misfit
end
