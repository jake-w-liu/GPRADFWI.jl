# run_hand_adjoint_baseline.jl
# Empirical reduced-domain baseline: AD vs finite-difference vs hand-coded adjoint.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using GPRADFWI
using Enzyme
using DelimitedFiles
using LinearAlgebra
using Printf
using Random
using Statistics

const DATADIR = joinpath(@__DIR__, "..", "..", "paper", "data")
mkpath(DATADIR)

struct SimpleTMConfig
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
    dt::Float64
    nt::Int
    src_ix::Int
    src_iy::Int
    rx_ix::Vector{Int}
    rx_iy::Int
    fc::Float64
end

function ricker_wavelet(nt::Int, dt::Float64, fc::Float64)
    t0 = 1.5 / fc
    w = Vector{Float64}(undef, nt)
    for n in 1:nt
        t = (n - 1) * dt
        a = pi * fc * (t - t0)
        w[n] = (1.0 - 2.0 * a^2) * exp(-a^2)
    end
    return w
end

"""
    run_forward_simple(eps_map, cfg, src_waveform; save_states=false)

Minimal non-dispersive TM FDTD used only for reduced-domain adjoint benchmarking.
Boundaries are fixed-value (PEC-like) by not updating edge nodes.
"""
function run_forward_simple(eps_map::Matrix{Float64},
                            cfg::SimpleTMConfig,
                            src_waveform::Vector{Float64};
                            save_states::Bool=false)
    nx, ny = cfg.nx, cfg.ny
    @assert size(eps_map) == (nx, ny)

    cb = cfg.dt ./ (eps0 .* eps_map)
    chx = cfg.dt / (mu0 * cfg.dx)
    chy = cfg.dt / (mu0 * cfg.dy)

    Ez = zeros(nx, ny)
    Hx = zeros(nx, ny)
    Hy = zeros(nx, ny)

    nrx = length(cfg.rx_ix)
    rec = zeros(cfg.nt, nrx)

    Ez_hist = save_states ? Vector{Matrix{Float64}}(undef, cfg.nt + 1) : Matrix{Float64}[]
    Hx_hist = save_states ? Vector{Matrix{Float64}}(undef, cfg.nt + 1) : Matrix{Float64}[]
    Hy_hist = save_states ? Vector{Matrix{Float64}}(undef, cfg.nt + 1) : Matrix{Float64}[]
    if save_states
        Ez_hist[1] = copy(Ez)
        Hx_hist[1] = copy(Hx)
        Hy_hist[1] = copy(Hy)
    end

    for n in 1:cfg.nt
        # H update from Ez^n
        for j in 1:ny-1, i in 1:nx
            Hx[i, j] -= chy * (Ez[i, j+1] - Ez[i, j])
        end
        for j in 1:ny, i in 1:nx-1
            Hy[i, j] += chx * (Ez[i+1, j] - Ez[i, j])
        end

        # Ez update from H^(n+1/2)
        for j in 2:ny-1, i in 2:nx-1
            curl_h = (Hy[i, j] - Hy[i-1, j]) / cfg.dx -
                     (Hx[i, j] - Hx[i, j-1]) / cfg.dy
            Ez[i, j] += cb[i, j] * curl_h
        end

        # Soft-source injection at Ez
        Ez[cfg.src_ix, cfg.src_iy] += cb[cfg.src_ix, cfg.src_iy] * src_waveform[n] / (cfg.dx * cfg.dy)

        for r in 1:nrx
            rec[n, r] = Ez[cfg.rx_ix[r], cfg.rx_iy]
        end

        if save_states
            Ez_hist[n+1] = copy(Ez)
            Hx_hist[n+1] = copy(Hx)
            Hy_hist[n+1] = copy(Hy)
        end
    end

    if save_states
        return rec, Ez_hist, Hx_hist, Hy_hist, cb
    end
    return rec
end

function build_reduced_problem()
    nx, ny = 48, 40
    dx = 0.01
    dy = dx
    cfl = 0.95 / sqrt(2)
    dt = cfl * dx / c0
    nt = 180
    fc = 250e6

    src_ix = nx ÷ 2
    src_iy = 6
    rx_ix = collect(6:3:(nx-5))
    rx_iy = 6

    cfg = SimpleTMConfig(nx, ny, dx, dy, dt, nt, src_ix, src_iy, rx_ix, rx_iy, fc)
    src_waveform = ricker_wavelet(nt, dt, fc)

    eps_true = 4.0 .* ones(nx, ny)
    eps_bg = copy(eps_true)

    anomaly_ix = nx ÷ 2
    anomaly_iy = ny ÷ 2 + 3
    anomaly_r = 4
    for j in 1:ny, i in 1:nx
        if (i - anomaly_ix)^2 + (j - anomaly_iy)^2 <= anomaly_r^2
            eps_true[i, j] = 8.5
        end
    end

    # Smaller inversion region keeps FD runtime manageable while still nontrivial.
    inv_half_w = 5
    inv_half_h = 5
    mask = falses(nx, ny)
    for j in (anomaly_iy-inv_half_h):(anomaly_iy+inv_half_h)
        for i in (anomaly_ix-inv_half_w):(anomaly_ix+inv_half_w)
            if 2 <= i <= nx-1 && 2 <= j <= ny-1
                mask[i, j] = true
            end
        end
    end
    param_cells = [CartesianIndex(i, j) for j in 1:ny for i in 1:nx if mask[i, j]]

    obs_data = run_forward_simple(eps_true, cfg, src_waveform)

    x0 = [eps_bg[idx] for idx in param_cells]

    return (
        cfg=cfg,
        src_waveform=src_waveform,
        obs_data=obs_data,
        eps_bg=eps_bg,
        param_cells=param_cells,
        x0=x0,
    )
end

function objective_from_params(x::Vector{Float64}, ctx)
    eps = copy(ctx.eps_bg)
    @inbounds for (k, idx) in enumerate(ctx.param_cells)
        eps[idx] = x[k]
    end
    syn = run_forward_simple(eps, ctx.cfg, ctx.src_waveform)
    misfit = 0.0
    @inbounds for j in axes(syn, 2), i in axes(syn, 1)
        r = syn[i, j] - ctx.obs_data[i, j]
        misfit += r * r
    end
    return 0.5 * misfit
end

# AD_WRAPPER_BEGIN
function ad_gradient_baseline(x::Vector{Float64}, ctx)
    f = xx -> objective_from_params(xx, ctx)
    dx = zeros(length(x))
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(f), Enzyme.Active, Enzyme.Duplicated(x, dx))
    return dx
end
# AD_WRAPPER_END

# HAND_ADJOINT_BEGIN
function hand_adjoint_gradient(x::Vector{Float64}, ctx)
    cfg = ctx.cfg
    nx, ny = cfg.nx, cfg.ny

    eps = copy(ctx.eps_bg)
    @inbounds for (k, idx) in enumerate(ctx.param_cells)
        eps[idx] = x[k]
    end

    syn, Ez_hist, Hx_hist, Hy_hist, cb = run_forward_simple(eps, cfg, ctx.src_waveform; save_states=true)
    residual = syn .- ctx.obs_data

    grad_cb = zeros(nx, ny)

    lambda_E_new = zeros(nx, ny)
    lambda_Hx_new = zeros(nx, ny)
    lambda_Hy_new = zeros(nx, ny)

    chx = cfg.dt / (mu0 * cfg.dx)
    chy = cfg.dt / (mu0 * cfg.dy)

    for n in cfg.nt:-1:1
        # Receiver contribution from J_n = 0.5 ||d_syn(n,:) - d_obs(n,:)||^2
        for (r, ix) in enumerate(cfg.rx_ix)
            lambda_E_new[ix, cfg.rx_iy] += residual[n, r]
        end

        # Reverse source injection: Ez_after = Ez_before + cb * src / (dx*dy)
        grad_cb[cfg.src_ix, cfg.src_iy] +=
            lambda_E_new[cfg.src_ix, cfg.src_iy] * ctx.src_waveform[n] / (cfg.dx * cfg.dy)

        # Reverse Ez update: Ez_before = Ez_old + cb .* curl(H_new)
        lambda_E_old = zeros(nx, ny)
        lambda_Hx_old = zeros(nx, ny)
        lambda_Hy_old = zeros(nx, ny)

        Hx_new = Hx_hist[n+1]
        Hy_new = Hy_hist[n+1]

        for j in 2:ny-1, i in 2:nx-1
            lam = lambda_E_new[i, j]
            if lam == 0.0
                continue
            end

            curl_h = (Hy_new[i, j] - Hy_new[i-1, j]) / cfg.dx -
                     (Hx_new[i, j] - Hx_new[i, j-1]) / cfg.dy

            lambda_E_old[i, j] += lam
            grad_cb[i, j] += lam * curl_h

            scale_x = lam * cb[i, j] / cfg.dx
            lambda_Hy_old[i, j] += scale_x
            lambda_Hy_old[i-1, j] -= scale_x

            scale_y = lam * cb[i, j] / cfg.dy
            lambda_Hx_old[i, j] -= scale_y
            lambda_Hx_old[i, j-1] += scale_y
        end

        # Add carry-over adjoint contributions from future to H_new
        lambda_Hx_old .+= lambda_Hx_new
        lambda_Hy_old .+= lambda_Hy_new

        # Reverse H update:
        # Hx_new = Hx_old - chy*(Ez_old[i,j+1] - Ez_old[i,j])
        for j in 1:ny-1, i in 1:nx
            lam = lambda_Hx_old[i, j]
            if lam == 0.0
                continue
            end
            lambda_E_old[i, j+1] -= chy * lam
            lambda_E_old[i, j] += chy * lam
        end

        # Hy_new = Hy_old + chx*(Ez_old[i+1,j] - Ez_old[i,j])
        for j in 1:ny, i in 1:nx-1
            lam = lambda_Hy_old[i, j]
            if lam == 0.0
                continue
            end
            lambda_E_old[i+1, j] += chx * lam
            lambda_E_old[i, j] -= chx * lam
        end

        lambda_E_new .= lambda_E_old
        lambda_Hx_new .= lambda_Hx_old
        lambda_Hy_new .= lambda_Hy_old
    end

    grad_eps_map = -grad_cb .* (cfg.dt ./ (eps0 .* (eps .^ 2)))

    grad = Vector{Float64}(undef, length(x))
    @inbounds for (k, idx) in enumerate(ctx.param_cells)
        grad[k] = grad_eps_map[idx]
    end

    return grad, syn
end
# HAND_ADJOINT_END

function rel_metrics(a::Vector{Float64}, b::Vector{Float64})
    rel = abs.(a .- b) ./ max.(abs.(b), 1e-12)
    return (
        max_rel=maximum(rel),
        mean_rel=mean(rel),
        l2_rel=norm(a .- b) / max(norm(b), 1e-12),
    )
end

function count_tagged_loc(path::String, tag_begin::String, tag_end::String)
    lines = readlines(path)
    begin_idx = findfirst(i -> strip(lines[i]) == tag_begin, eachindex(lines))
    end_idx = findfirst(i -> strip(lines[i]) == tag_end, eachindex(lines))
    if begin_idx === nothing || end_idx === nothing || end_idx <= begin_idx
        return 0
    end
    return end_idx - begin_idx - 1
end

function main()
    Random.seed!(123)

    println("Building reduced hand-adjoint benchmark problem...")
    ctx = build_reduced_problem()
    n_params = length(ctx.x0)
    @printf("  Domain: %d x %d, nt=%d\n", ctx.cfg.nx, ctx.cfg.ny, ctx.cfg.nt)
    @printf("  Receivers: %d, inversion parameters: %d\n", length(ctx.cfg.rx_ix), n_params)
    flush(stdout)

    # Warmup
    println("Warmup runs...")
    objective_from_params(ctx.x0, ctx)
    ad_gradient_baseline(ctx.x0, ctx)
    hand_adjoint_gradient(ctx.x0, ctx)
    flush(stdout)

    println("Computing gradients...")
    g_ad = ad_gradient_baseline(ctx.x0, ctx)
    g_fd = fd_gradient(x -> objective_from_params(x, ctx), ctx.x0; h=1e-5)
    g_hand, syn = hand_adjoint_gradient(ctx.x0, ctx)
    flush(stdout)

    m_ad_hand = rel_metrics(g_ad, g_hand)
    m_fd_hand = rel_metrics(g_fd, g_hand)
    m_ad_fd = rel_metrics(g_ad, g_fd)

    # Timing (steady-state, already warmed up)
    t_forward = mean([@elapsed run_forward_simple(ctx.eps_bg, ctx.cfg, ctx.src_waveform) for _ in 1:6])
    t_ad = mean([@elapsed ad_gradient_baseline(ctx.x0, ctx) for _ in 1:4])
    t_hand = mean([@elapsed hand_adjoint_gradient(ctx.x0, ctx) for _ in 1:4])
    t_fd = @elapsed fd_gradient(x -> objective_from_params(x, ctx), ctx.x0; h=1e-5)

    # Implementation footprint (local to this benchmark script)
    this_file = @__FILE__
    loc_hand = count_tagged_loc(this_file, "# HAND_ADJOINT_BEGIN", "# HAND_ADJOINT_END")
    loc_ad = count_tagged_loc(this_file, "# AD_WRAPPER_BEGIN", "# AD_WRAPPER_END")

    # Save per-parameter gradient comparison
    grad_file = joinpath(DATADIR, "hand_adjoint_gradient_comparison.csv")
    open(grad_file, "w") do io
        write(io, "# Reduced-domain baseline: AD vs FD vs hand adjoint gradient\n")
        write(io, "param_idx,gradient_ad,gradient_fd,gradient_hand_adj,rel_err_ad_vs_hand,rel_err_fd_vs_hand\n")
        for k in eachindex(g_hand)
            rel_ad = abs(g_ad[k] - g_hand[k]) / max(abs(g_hand[k]), 1e-12)
            rel_fd = abs(g_fd[k] - g_hand[k]) / max(abs(g_hand[k]), 1e-12)
            @printf(io, "%d,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                    k, g_ad[k], g_fd[k], g_hand[k], rel_ad, rel_fd)
        end
    end

    timing_file = joinpath(DATADIR, "hand_adjoint_timing.csv")
    open(timing_file, "w") do io
        write(io, "# Reduced-domain timing: gradient methods\n")
        write(io, "operation,wallclock_s,relative_to_forward\n")
        @printf(io, "forward,%.6f,1.0\n", t_forward)
        @printf(io, "ad_gradient,%.6f,%.6f\n", t_ad, t_ad / t_forward)
        @printf(io, "hand_adjoint_gradient,%.6f,%.6f\n", t_hand, t_hand / t_forward)
        @printf(io, "fd_gradient,%.6f,%.6f\n", t_fd, t_fd / t_forward)
    end

    summary_file = joinpath(DATADIR, "hand_adjoint_summary.csv")
    open(summary_file, "w") do io
        write(io, "# Reduced-domain hand-adjoint baseline summary\n")
        write(io, "metric_name,value_si\n")
        @printf(io, "n_params,%d\n", n_params)
        @printf(io, "misfit_initial,%.12e\n", 0.5 * sum((syn .- ctx.obs_data) .^ 2))
        @printf(io, "max_relerr_ad_vs_hand,%.12e\n", m_ad_hand.max_rel)
        @printf(io, "mean_relerr_ad_vs_hand,%.12e\n", m_ad_hand.mean_rel)
        @printf(io, "l2_relerr_ad_vs_hand,%.12e\n", m_ad_hand.l2_rel)
        @printf(io, "max_relerr_fd_vs_hand,%.12e\n", m_fd_hand.max_rel)
        @printf(io, "mean_relerr_fd_vs_hand,%.12e\n", m_fd_hand.mean_rel)
        @printf(io, "l2_relerr_fd_vs_hand,%.12e\n", m_fd_hand.l2_rel)
        @printf(io, "max_relerr_ad_vs_fd,%.12e\n", m_ad_fd.max_rel)
        @printf(io, "mean_relerr_ad_vs_fd,%.12e\n", m_ad_fd.mean_rel)
        @printf(io, "l2_relerr_ad_vs_fd,%.12e\n", m_ad_fd.l2_rel)
        @printf(io, "time_forward_s,%.6f\n", t_forward)
        @printf(io, "time_ad_gradient_s,%.6f\n", t_ad)
        @printf(io, "time_hand_adjoint_s,%.6f\n", t_hand)
        @printf(io, "time_fd_gradient_s,%.6f\n", t_fd)
        @printf(io, "ad_vs_hand_speed_ratio,%.6f\n", t_ad / t_hand)
        @printf(io, "hand_vs_fd_speed_ratio,%.6f\n", t_fd / t_hand)
    end

    footprint_file = joinpath(DATADIR, "hand_adjoint_footprint.csv")
    open(footprint_file, "w") do io
        write(io, "# Implementation footprint in reduced-domain baseline script\n")
        write(io, "method,loc_in_experiment,new_functions,new_modules,notes\n")
        @printf(io, "ad_gradient,%d,%d,%d,%s\n",
                loc_ad, 1, 0,
                "Single autodiff call wrapper; reuses forward solver")
        @printf(io, "hand_adjoint,%d,%d,%d,%s\n",
                loc_hand, 1, 0,
                "Explicit reverse-time adjoint for TM update equations")
    end

    println("Saved: $grad_file")
    println("Saved: $timing_file")
    println("Saved: $summary_file")
    println("Saved: $footprint_file")

    println()
    println("Gradient agreement:")
    @printf("  AD vs hand adjoint: max=%.3e, mean=%.3e, L2=%.3e\n",
            m_ad_hand.max_rel, m_ad_hand.mean_rel, m_ad_hand.l2_rel)
    @printf("  FD vs hand adjoint: max=%.3e, mean=%.3e, L2=%.3e\n",
            m_fd_hand.max_rel, m_fd_hand.mean_rel, m_fd_hand.l2_rel)

    println("Timing (steady-state):")
    @printf("  Forward:      %.4f s\n", t_forward)
    @printf("  AD gradient:  %.4f s (%.1fx forward)\n", t_ad, t_ad / t_forward)
    @printf("  Hand adjoint: %.4f s (%.1fx forward)\n", t_hand, t_hand / t_forward)
    @printf("  FD gradient:  %.4f s (%.1fx forward)\n", t_fd, t_fd / t_forward)

    println("\n=== Hand-adjoint baseline complete ===")
end

main()
