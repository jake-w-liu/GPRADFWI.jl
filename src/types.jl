# types.jl — Data structures for 2D TM-mode FDTD GPR simulation
#
# Time convention: e^{+iωt}
# 2D TM mode: Ez (out-of-plane), Hx, Hy (in-plane)
# Yee grid: Ez at integer nodes, H at half-integer nodes

"""
    DebyeMedium

Single-pole Debye dispersive medium parameters.

    ε(ω) = ε∞ + Δε / (1 + iωτ) + σ/(iωε₀)

Note: using e^{+iωt} convention, the denominator is (1 + iωτ).
"""
struct DebyeMedium
    eps_inf::Float64   # high-frequency relative permittivity
    deps::Float64      # Δε = εs - ε∞ (relaxation strength)
    tau::Float64       # relaxation time [s]
    sigma::Float64     # static conductivity [S/m]
    mu_r::Float64      # relative permeability (usually 1.0)
end

function DebyeMedium(; eps_inf=1.0, deps=0.0, tau=0.0, sigma=0.0, mu_r=1.0)
    return DebyeMedium(eps_inf, deps, tau, sigma, mu_r)
end

"""
Non-dispersive medium (Debye with deps=0, tau=0).
"""
function SimpleMedium(; eps_r=1.0, sigma=0.0, mu_r=1.0)
    return DebyeMedium(eps_r, 0.0, 0.0, sigma, mu_r)
end

"""
    CPMLParams

Convolutional PML parameters for absorbing boundaries.
"""
struct CPMLParams
    npml::Int          # number of PML cells
    order::Int         # polynomial grading order
    kappa_max::Float64 # maximum stretching factor
    alpha_max::Float64 # maximum CFS alpha
    sigma_fac::Float64 # sigma scaling factor relative to optimal
end

function CPMLParams(; npml=10, order=3, kappa_max=11.0, alpha_max=0.05, sigma_fac=1.2)
    return CPMLParams(npml, order, kappa_max, alpha_max, sigma_fac)
end

"""
    SourceConfig

GPR source configuration (Ricker wavelet point source).
"""
struct SourceConfig
    fc::Float64        # center frequency [Hz]
    ix::Int            # source x-index on grid
    iy::Int            # source y-index on grid
    t0::Float64        # time delay [s] (peak of Ricker wavelet)
end

"""
    FDTDConfig

Complete configuration for 2D TM-mode FDTD simulation.
"""
struct FDTDConfig
    # Grid
    nx::Int            # number of cells in x
    ny::Int            # number of cells in y
    dx::Float64        # cell size x [m]
    dy::Float64        # cell size y [m]
    dt::Float64        # time step [s]
    nt::Int            # number of time steps

    # PML
    cpml::CPMLParams

    # Source
    source::SourceConfig

    # Receivers: (ix, iy) pairs
    rx_ix::Vector{Int}
    rx_iy::Vector{Int}
end

"""
    GPRSetup

Complete GPR simulation setup including domain geometry and material model.
"""
struct GPRSetup
    config::FDTDConfig
    # Material maps (nx × ny): parameterize by eps_inf, deps, tau, sigma
    eps_inf_map::Matrix{Float64}
    deps_map::Matrix{Float64}
    tau_map::Matrix{Float64}
    sigma_map::Matrix{Float64}
end

"""
    FWIResult

Output from full-waveform inversion.
"""
struct FWIResult
    eps_inf_est::Matrix{Float64}
    deps_est::Matrix{Float64}
    sigma_est::Matrix{Float64}
    loss_history::Vector{Float64}
    grad_norm_history::Vector{Float64}
    loss_data_history::Vector{Float64}
    loss_reg_eps_history::Vector{Float64}
    loss_reg_sigma_history::Vector{Float64}
    step_alpha_history::Vector{Float64}
    line_search_backtracks::Vector{Int}
    n_iter::Int
end

"""
    create_config(; nx, ny, dx, fc, npml, src_ix, src_iy, rx_iy, rx_ix_list, nt)

Convenience constructor for FDTDConfig with automatic dt (CFL) and source delay.
"""
function create_config(;
    nx::Int,
    ny::Int,
    dx::Float64,
    fc::Float64,
    npml::Int = 10,
    src_ix::Int,
    src_iy::Int,
    rx_iy::Int,
    rx_ix_list::Vector{Int},
    nt::Int = 0,
)
    dy = dx
    # CFL condition for 2D: dt ≤ 1/(c0 * sqrt(1/dx² + 1/dy²))
    dt = 0.99 / (c0 * sqrt(1.0 / dx^2 + 1.0 / dy^2))

    # Source delay: 1.5 periods of Ricker wavelet
    t0 = 1.5 / fc

    # Auto time steps if not specified: enough for wave to traverse domain twice
    if nt == 0
        domain_size = max(nx * dx, ny * dy)
        nt = round(Int, 3.0 * domain_size / (c0 * dt))
        nt = max(nt, round(Int, 6.0 / (fc * dt)))  # at least 6 periods
    end

    source = SourceConfig(fc, src_ix, src_iy, t0)
    cpml = CPMLParams(npml=npml)
    rx_iy_vec = fill(rx_iy, length(rx_ix_list))

    return FDTDConfig(nx, ny, dx, dy, dt, nt, cpml, source, rx_ix_list, rx_iy_vec)
end
