# fdtd.jl — 2D TM-mode FDTD forward solver with Debye ADE and CPML
#
# Time convention: e^{+iωt}
# Field components: Ez (z-polarized), Hx, Hy (in-plane)
#
# Yee grid layout (2D):
#   Ez(i,j) lives at integer node (i, j)
#   Hx(i,j) lives at (i, j+1/2)  — between Ez(i,j) and Ez(i,j+1)
#   Hy(i,j) lives at (i+1/2, j)  — between Ez(i,j) and Ez(i+1,j)
#
# Update order per time step n:
#   1. Update Hx, Hy from Ez  (H^{n+1/2} from E^n)
#   2. Update CPML ψ for H
#   3. Update Ez from Hx, Hy  (E^{n+1} from H^{n+1/2}), with Debye ADE
#   4. Update CPML ψ for E
#   5. Add source
#   6. Record receivers

"""
    run_forward!(config, eps_inf, deps, tau, sigma, src_waveform)

Run the 2D TM-mode FDTD forward simulation. Returns recorded Ez at receivers.

# Arguments
- `config::FDTDConfig`: simulation configuration
- `eps_inf, deps, tau, sigma`: material parameter maps (nx × ny)
- `src_waveform::Vector{Float64}`: pre-computed source time series

# Returns
- `rec_data::Matrix{Float64}`: recorded Ez (nt × nrx)
"""
function run_forward!(config::FDTDConfig,
                      eps_inf::Matrix{Float64},
                      deps::Matrix{Float64},
                      tau::Matrix{Float64},
                      sigma::Matrix{Float64},
                      src_waveform::Vector{Float64})
    nx = config.nx
    ny = config.ny
    dt = config.dt
    dx = config.dx
    dy = config.dy
    nt = config.nt

    @assert size(eps_inf) == (nx, ny) "eps_inf must be $nx × $ny"
    @assert size(deps) == (nx, ny) "deps must be $nx × $ny"
    @assert length(src_waveform) == nt "source length must be $nt"

    # Initialize fields
    Ez = zeros(nx, ny)
    Hx = zeros(nx, ny)
    Hy = zeros(nx, ny)
    Pz = zeros(nx, ny)  # Debye polarization

    # Pre-compute Debye ADE coefficients
    dcoeffs = init_debye_coeffs(eps_inf, deps, tau, sigma, dt, nx, ny)

    # H-field update coefficient: dt/(μ₀ dx)
    ch_dt_dx = dt / (mu0 * dx)
    ch_dt_dy = dt / (mu0 * dy)

    # Initialize CPML
    cpml = init_cpml(config)

    # Receiver data
    nrx = length(config.rx_ix)
    rec_data = zeros(nt, nrx)

    # Source indices
    si = config.source.ix
    sj = config.source.iy

    # ── Time-stepping loop ─────────────────────────────────────────────
    for n in 1:nt
        # 1. Update H-fields: H^{n+1/2} from E^n
        _update_H!(Hx, Hy, Ez, ch_dt_dx, ch_dt_dy, cpml, nx, ny)

        # 2. Update E-field (Ez) with Debye ADE: E^{n+1} from H^{n+1/2}
        _update_E_debye!(Ez, Hx, Hy, Pz, dcoeffs, cpml, dt, dx, dy, nx, ny)

        # 3. Add source (soft source: additive injection into Ez)
        Ez[si, sj] += dcoeffs.cb[si, sj] * src_waveform[n] / (dx * dy)

        # 4. Record receivers
        for r in 1:nrx
            rec_data[n, r] = Ez[config.rx_ix[r], config.rx_iy[r]]
        end
    end

    return rec_data
end

"""
Update H-fields with CPML.
Hx^{n+1/2} = Hx^{n-1/2} - dt/(μ₀) * ∂Ez/∂y
Hy^{n+1/2} = Hy^{n-1/2} + dt/(μ₀) * ∂Ez/∂x
"""
function _update_H!(Hx, Hy, Ez, ch_dt_dx, ch_dt_dy, cpml, nx, ny)
    # Hx update: Hx(i,j) uses Ez(i,j) and Ez(i,j+1)
    for j in 1:ny-1, i in 1:nx
        dEz = Ez[i, j+1] - Ez[i, j]  # raw Yee-grid difference (normalization via ch_dt_dy)
        Hx[i, j] -= ch_dt_dy * (dEz / cpml.kappa_hy[j])

        # CPML y-boundaries
        if j <= size(cpml.psi_hxy_y1, 2)
            cpml.psi_hxy_y1[i, j] = cpml.bh_y[j] * cpml.psi_hxy_y1[i, j] +
                                     cpml.ch_y[j] * dEz
            Hx[i, j] -= ch_dt_dy * cpml.psi_hxy_y1[i, j]
        end
        jj = j - (ny - size(cpml.psi_hxy_y2, 2))
        if jj >= 1 && jj <= size(cpml.psi_hxy_y2, 2)
            cpml.psi_hxy_y2[i, jj] = cpml.bh_y[j] * cpml.psi_hxy_y2[i, jj] +
                                       cpml.ch_y[j] * dEz
            Hx[i, j] -= ch_dt_dy * cpml.psi_hxy_y2[i, jj]
        end
    end

    # Hy update: Hy(i,j) uses Ez(i+1,j) and Ez(i,j)
    for j in 1:ny, i in 1:nx-1
        dEz = Ez[i+1, j] - Ez[i, j]
        Hy[i, j] += ch_dt_dx * (dEz / cpml.kappa_hx[i])

        # CPML x-boundaries
        if i <= size(cpml.psi_hyx_x1, 1)
            cpml.psi_hyx_x1[i, j] = cpml.bh_x[i] * cpml.psi_hyx_x1[i, j] +
                                      cpml.ch_x[i] * dEz
            Hy[i, j] += ch_dt_dx * cpml.psi_hyx_x1[i, j]
        end
        ii = i - (nx - size(cpml.psi_hyx_x2, 1))
        if ii >= 1 && ii <= size(cpml.psi_hyx_x2, 1)
            cpml.psi_hyx_x2[ii, j] = cpml.bh_x[i] * cpml.psi_hyx_x2[ii, j] +
                                       cpml.ch_x[i] * dEz
            Hy[i, j] += ch_dt_dx * cpml.psi_hyx_x2[ii, j]
        end
    end
end

"""
Update Ez with Debye ADE and CPML.

Combined update:
  E^{n+1} = ca * E^n + cb * (∂Hy/∂x - ∂Hx/∂y) + cp * P^n
  P^{n+1} = c1 * P^n + c2 * E^{n+1}
"""
function _update_E_debye!(Ez, Hx, Hy, Pz, dc::DebyeCoeffs, cpml, dt, dx, dy, nx, ny)
    for j in 2:ny-1, i in 2:nx-1
        # Curl H (finite differences on Yee grid)
        dHy = Hy[i, j] - Hy[i-1, j]
        dHx = Hx[i, j] - Hx[i, j-1]
        curl_H = dHy / (cpml.kappa_ex[i] * dx) - dHx / (cpml.kappa_ey[j] * dy)

        # CPML corrections for curl_H
        curl_H_cpml = 0.0

        # x-direction CPML
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

        # y-direction CPML
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

        # Store old polarization and E for ADE
        Pz_old = Pz[i, j]
        Ez_old = Ez[i, j]

        # Combined E-field update with Debye
        Ez[i, j] = dc.ca[i, j] * Ez_old +
                    dc.cb[i, j] * (curl_H + curl_H_cpml) +
                    dc.cp[i, j] * Pz_old

        # Update Debye polarization
        Pz[i, j] = dc.c1[i, j] * Pz_old + dc.c2[i, j] * Ez[i, j]
    end
end

# ── Forward solver with field snapshots (visualization only) ──────────

"""
    run_forward_snapshots(config, eps_inf, deps, tau, sigma, src_waveform, snap_steps)

Run the 2D TM-mode FDTD forward simulation and capture Ez field snapshots
at specified time steps. NOT used for AD — purely for visualization.

# Arguments
- `config::FDTDConfig`: simulation configuration
- `eps_inf, deps, tau, sigma`: material parameter maps (nx × ny)
- `src_waveform::Vector{Float64}`: pre-computed source time series
- `snap_steps::Vector{Int}`: time-step indices at which to capture Ez snapshots

# Returns
- `rec_data::Matrix{Float64}`: recorded Ez at receivers (nt × nrx)
- `snapshots::Vector{Matrix{Float64}}`: Ez snapshots in order of snap_steps (each nx × ny)
"""
function run_forward_snapshots(config::FDTDConfig,
                               eps_inf::Matrix{Float64},
                               deps::Matrix{Float64},
                               tau::Matrix{Float64},
                               sigma::Matrix{Float64},
                               src_waveform::Vector{Float64},
                               snap_steps::Vector{Int})
    nx = config.nx
    ny = config.ny
    dt = config.dt
    dx = config.dx
    dy = config.dy
    nt = config.nt

    @assert size(eps_inf) == (nx, ny) "eps_inf must be $nx × $ny"
    @assert size(deps) == (nx, ny) "deps must be $nx × $ny"
    @assert length(src_waveform) == nt "source length must be $nt"
    @assert all(1 .<= snap_steps .<= nt) "snap_steps must be in [1, nt]"

    # Initialize fields
    Ez = zeros(nx, ny)
    Hx = zeros(nx, ny)
    Hy = zeros(nx, ny)
    Pz = zeros(nx, ny)

    # Pre-compute Debye ADE coefficients
    dcoeffs = init_debye_coeffs(eps_inf, deps, tau, sigma, dt, nx, ny)

    # H-field update coefficient
    ch_dt_dx = dt / (mu0 * dx)
    ch_dt_dy = dt / (mu0 * dy)

    # Initialize CPML
    cpml = init_cpml(config)

    # Receiver data
    nrx = length(config.rx_ix)
    rec_data = zeros(nt, nrx)

    # Source indices
    si = config.source.ix
    sj = config.source.iy

    # Snapshot storage
    snap_set = Set(snap_steps)
    snapshots = Vector{Matrix{Float64}}()
    snap_order = Dict(s => k for (k, s) in enumerate(snap_steps))

    # Pre-allocate in order
    for _ in 1:length(snap_steps)
        push!(snapshots, Matrix{Float64}(undef, 0, 0))
    end

    # ── Time-stepping loop ─────────────────────────────────────────────
    for n in 1:nt
        _update_H!(Hx, Hy, Ez, ch_dt_dx, ch_dt_dy, cpml, nx, ny)
        _update_E_debye!(Ez, Hx, Hy, Pz, dcoeffs, cpml, dt, dx, dy, nx, ny)
        Ez[si, sj] += dcoeffs.cb[si, sj] * src_waveform[n] / (dx * dy)

        for r in 1:nrx
            rec_data[n, r] = Ez[config.rx_ix[r], config.rx_iy[r]]
        end

        if n in snap_set
            snapshots[snap_order[n]] = copy(Ez)
        end
    end

    return rec_data, snapshots
end

# ── Differentiable forward solver for AD ──────────────────────────────

"""
    forward_misfit(params_flat, config, obs_data, src_waveform,
                   eps_inf_bg, tau_map, param_mask, param_type)

Differentiable objective function: runs forward simulation and computes
L2 misfit against observed data. Designed for Enzyme.jl reverse-mode AD.

`params_flat` is the vector of parameters being optimized.
`param_type` ∈ {:eps_inf, :sigma, :both} selects which parameters to invert.
`param_mask` is a BitMatrix indicating which cells are inverted.
"""
function forward_misfit(params_flat::Vector{Float64},
                        config::FDTDConfig,
                        obs_data::Matrix{Float64},
                        src_waveform::Vector{Float64},
                        eps_inf_bg::Matrix{Float64},
                        deps_map::Matrix{Float64},
                        tau_map::Matrix{Float64},
                        sigma_bg::Matrix{Float64},
                        param_mask::BitMatrix,
                        param_type::Symbol)
    nx = config.nx
    ny = config.ny

    # Unpack parameters into material maps
    eps_inf = copy(eps_inf_bg)
    sigma = copy(sigma_bg)

    idx = 1
    for j in 1:ny, i in 1:nx
        if param_mask[i, j]
            if param_type == :eps_inf || param_type == :both
                eps_inf[i, j] = params_flat[idx]
                idx += 1
            end
            if param_type == :sigma || param_type == :both
                sigma[i, j] = params_flat[idx]
                idx += 1
            end
        end
    end

    # Run forward simulation
    syn_data = run_forward!(config, eps_inf, deps_map, tau_map, sigma, src_waveform)

    # L2 misfit: 0.5 * ||d_syn - d_obs||²
    misfit = 0.0
    for j in 1:size(obs_data, 2), i in 1:size(obs_data, 1)
        r = syn_data[i, j] - obs_data[i, j]
        misfit += r * r
    end
    return 0.5 * misfit
end

"""
    compute_misfit(syn_data, obs_data)

L2 norm misfit: 0.5 * ||syn - obs||²
"""
function compute_misfit(syn_data::Matrix{Float64}, obs_data::Matrix{Float64})
    misfit = 0.0
    for j in axes(obs_data, 2), i in axes(obs_data, 1)
        r = syn_data[i, j] - obs_data[i, j]
        misfit += r * r
    end
    return 0.5 * misfit
end
