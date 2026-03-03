# cpml.jl — Convolutional Perfectly Matched Layer (CPML) for 2D TM-mode FDTD
#
# CPML stretching: s_i = κ_i + σ_i / (α_i + iω)
# Implemented via auxiliary recursive convolution variables ψ.

"""
    CPMLData

Pre-computed CPML coefficients and auxiliary variables for 2D boundaries.
Stored separately for E-field and H-field updates on each boundary.
"""
mutable struct CPMLData
    # Coefficients for x-boundaries (applied along x)
    be_x::Vector{Float64}   # b coefficient for E update, x-direction
    ce_x::Vector{Float64}   # c coefficient for E update, x-direction
    bh_x::Vector{Float64}   # b coefficient for H update, x-direction
    ch_x::Vector{Float64}   # c coefficient for H update, x-direction
    kappa_ex::Vector{Float64}
    kappa_hx::Vector{Float64}

    # Coefficients for y-boundaries (applied along y)
    be_y::Vector{Float64}
    ce_y::Vector{Float64}
    bh_y::Vector{Float64}
    ch_y::Vector{Float64}
    kappa_ey::Vector{Float64}
    kappa_hy::Vector{Float64}

    # Auxiliary ψ fields (only in PML regions)
    # x-boundaries: ψ for ∂/∂x terms
    psi_ezx_x1::Matrix{Float64}  # left PML, for Ez update (∂Hy/∂x)
    psi_ezx_x2::Matrix{Float64}  # right PML
    psi_hyx_x1::Matrix{Float64}  # left PML, for Hy update (∂Ez/∂x)
    psi_hyx_x2::Matrix{Float64}  # right PML

    # y-boundaries: ψ for ∂/∂y terms
    psi_ezy_y1::Matrix{Float64}  # bottom PML, for Ez update (∂Hx/∂y)
    psi_ezy_y2::Matrix{Float64}  # top PML
    psi_hxy_y1::Matrix{Float64}  # bottom PML, for Hx update (∂Ez/∂y)
    psi_hxy_y2::Matrix{Float64}  # top PML
end

"""
    init_cpml(config::FDTDConfig) -> CPMLData

Initialize CPML coefficients and zero-valued auxiliary fields.
Polynomial grading of σ, κ, α from domain interior to PML edge.
"""
function init_cpml(config::FDTDConfig)
    npml = config.cpml.npml
    nx = config.nx
    ny = config.ny
    dt = config.dt
    dx = config.dx
    dy = config.dy
    order = config.cpml.order
    kappa_max = config.cpml.kappa_max
    alpha_max = config.cpml.alpha_max
    sigma_fac = config.cpml.sigma_fac

    # Optimal sigma for the given PML thickness
    sigma_opt_x = sigma_fac * (order + 1) / (2.0 * eta0 * dx)
    sigma_opt_y = sigma_fac * (order + 1) / (2.0 * eta0 * dy)

    # Pre-allocate coefficient vectors (full grid length)
    be_x  = zeros(nx); ce_x  = zeros(nx); kappa_ex = ones(nx)
    bh_x  = zeros(nx); ch_x  = zeros(nx); kappa_hx = ones(nx)
    be_y  = zeros(ny); ce_y  = zeros(ny); kappa_ey = ones(ny)
    bh_y  = zeros(ny); ch_y  = zeros(ny); kappa_hy = ones(ny)

    # Fill coefficients for x-direction PML
    for i in 1:nx
        # E-node positions (integer grid)
        de = _pml_distance_e(i, nx, npml)
        if de >= 0.0
            se, ke, ae = _pml_profile(de, npml, order, sigma_opt_x, kappa_max, alpha_max)
            be_x[i] = exp(-(se / ke + ae) * dt / eps0)
            ce_x[i] = se / (se * ke + ke^2 * ae) * (be_x[i] - 1.0)
            kappa_ex[i] = ke
        end
        # H-node positions (half-integer grid)
        dh = _pml_distance_h(i, nx, npml)
        if dh >= 0.0
            sh, kh, ah = _pml_profile(dh, npml, order, sigma_opt_x, kappa_max, alpha_max)
            bh_x[i] = exp(-(sh / kh + ah) * dt / eps0)
            ch_x[i] = sh / (sh * kh + kh^2 * ah) * (bh_x[i] - 1.0)
            kappa_hx[i] = kh
        end
    end

    # Fill coefficients for y-direction PML
    for j in 1:ny
        de = _pml_distance_e(j, ny, npml)
        if de >= 0.0
            se, ke, ae = _pml_profile(de, npml, order, sigma_opt_y, kappa_max, alpha_max)
            be_y[j] = exp(-(se / ke + ae) * dt / eps0)
            ce_y[j] = se / (se * ke + ke^2 * ae) * (be_y[j] - 1.0)
            kappa_ey[j] = ke
        end
        dh = _pml_distance_h(j, ny, npml)
        if dh >= 0.0
            sh, kh, ah = _pml_profile(dh, npml, order, sigma_opt_y, kappa_max, alpha_max)
            bh_y[j] = exp(-(sh / kh + ah) * dt / eps0)
            ch_y[j] = sh / (sh * kh + kh^2 * ah) * (bh_y[j] - 1.0)
            kappa_hy[j] = kh
        end
    end

    # Auxiliary ψ fields (sized for PML regions only)
    psi_ezx_x1 = zeros(npml, ny)
    psi_ezx_x2 = zeros(npml, ny)
    psi_hyx_x1 = zeros(npml, ny)
    psi_hyx_x2 = zeros(npml, ny)

    psi_ezy_y1 = zeros(nx, npml)
    psi_ezy_y2 = zeros(nx, npml)
    psi_hxy_y1 = zeros(nx, npml)
    psi_hxy_y2 = zeros(nx, npml)

    return CPMLData(
        be_x, ce_x, bh_x, ch_x, kappa_ex, kappa_hx,
        be_y, ce_y, bh_y, ch_y, kappa_ey, kappa_hy,
        psi_ezx_x1, psi_ezx_x2, psi_hyx_x1, psi_hyx_x2,
        psi_ezy_y1, psi_ezy_y2, psi_hxy_y1, psi_hxy_y2,
    )
end

"""
Normalized distance from boundary for E-nodes (integer positions).
Returns -1.0 if not in PML.
"""
function _pml_distance_e(i::Int, n::Int, npml::Int)
    if i <= npml
        return (npml - i + 1.0) / npml  # left/bottom PML
    elseif i > n - npml
        return (i - (n - npml)) / npml   # right/top PML
    else
        return -1.0
    end
end

"""
Normalized distance from boundary for H-nodes (half-integer positions i+0.5).
Returns -1.0 if not in PML.
"""
function _pml_distance_h(i::Int, n::Int, npml::Int)
    if i <= npml
        return (npml - i + 0.5) / npml
    elseif i >= n - npml
        return (i - (n - npml) + 0.5) / npml
    else
        return -1.0
    end
end

"""
Polynomial PML profile: σ(d), κ(d), α(d) for normalized distance d ∈ [0,1].
"""
function _pml_profile(d::Float64, npml::Int, order::Int,
                      sigma_opt::Float64, kappa_max::Float64, alpha_max::Float64)
    sigma = sigma_opt * d^order
    kappa = 1.0 + (kappa_max - 1.0) * d^order
    alpha = alpha_max * (1.0 - d)  # α tapers to zero at outer edge
    return sigma, kappa, alpha
end
