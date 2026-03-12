# run_revision_validations.jl
# Revision artifacts:
#   R2: FD step-size sweep for dispersive gradient verification
#   R3: Full-physics (Debye+CPML) AD-vs-FD comparator under matched budget

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using DelimitedFiles
using Enzyme
using GPRADFWI
using LinearAlgebra
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "revision_reduced_common.jl"))

const DATADIR = joinpath(@__DIR__, "..", "..", "paper", "data")
mkpath(DATADIR)

function build_gradient_validation_case()
    nx = 60
    ny = 50
    dx = 0.01
    fc = 300e6
    npml = 8

    rx_list = collect((npml+2):3:(nx-npml-2))
    rx_y = npml + 5
    src_x = nx ÷ 2
    src_y = rx_y

    config = create_config(
        nx=nx, ny=ny, dx=dx, fc=fc, npml=npml,
        src_ix=src_x, src_iy=src_y,
        rx_iy=rx_y, rx_ix_list=rx_list,
        nt=300,
    )

    eps_inf_true = 3.0 * ones(nx, ny)
    sigma_true = 0.001 * ones(nx, ny)
    deps = 2.0 * ones(nx, ny)
    tau = 0.4e-9 * ones(nx, ny)

    anomaly_i = nx ÷ 2
    anomaly_j = ny ÷ 2
    anomaly_r = 3
    for j in 1:ny, i in 1:nx
        if (i - anomaly_i)^2 + (j - anomaly_j)^2 <= anomaly_r^2
            eps_inf_true[i, j] = 6.0
        end
    end

    src = create_source(config)
    obs_data = run_forward!(config, eps_inf_true, deps, tau, sigma_true, src)

    param_mask = falses(nx, ny)
    for j in (anomaly_j-6):(anomaly_j+6), i in (anomaly_i-6):(anomaly_i+6)
        if 1 <= i <= nx && 1 <= j <= ny
            param_mask[i, j] = true
        end
    end

    eps_bg = 3.0 * ones(nx, ny)
    sigma_bg = 0.001 * ones(nx, ny)

    x0 = Float64[]
    for j in 1:ny, i in 1:nx
        if param_mask[i, j]
            push!(x0, eps_bg[i, j])
        end
    end

    return (
        config=config,
        obs_data=obs_data,
        src=src,
        eps_bg=eps_bg,
        deps=deps,
        tau=tau,
        sigma_bg=sigma_bg,
        param_mask=param_mask,
        x0=x0,
    )
end

function run_gradient_stepsweep()
    println("Running R2: gradient FD step-size sweep...")
    ctx = build_gradient_validation_case()

    function obj(x_flat)
        return forward_misfit(
            x_flat, ctx.config, ctx.obs_data, ctx.src,
            ctx.eps_bg, ctx.deps, ctx.tau, ctx.sigma_bg,
            ctx.param_mask, :eps_inf,
        )
    end

    grad_ad = zeros(length(ctx.x0))
    Enzyme.autodiff(Enzyme.Reverse, forward_misfit, Enzyme.Active,
                    Enzyme.Duplicated(ctx.x0, grad_ad),
                    Enzyme.Const(ctx.config),
                    Enzyme.Const(ctx.obs_data),
                    Enzyme.Const(ctx.src),
                    Enzyme.Const(ctx.eps_bg),
                    Enzyme.Const(ctx.deps),
                    Enzyme.Const(ctx.tau),
                    Enzyme.Const(ctx.sigma_bg),
                    Enzyme.Const(ctx.param_mask),
                    Enzyme.Const(:eps_inf))

    h_values = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]

    rows = NamedTuple[]
    best_idx = 1
    best_max_rel = Inf
    for (k, h) in enumerate(h_values)
        t_fd = @elapsed grad_fd = fd_gradient(obj, ctx.x0; h=h)
        rel = abs.(grad_ad .- grad_fd) ./ max.(abs.(grad_fd), 1e-30)
        max_rel = maximum(rel)
        mean_rel = mean(rel)
        l2_rel = norm(grad_ad - grad_fd) / max(norm(grad_fd), 1e-30)
        push!(rows, (
            h=h,
            fd_time_s=t_fd,
            max_rel=max_rel,
            mean_rel=mean_rel,
            l2_rel=l2_rel,
        ))
        if max_rel < best_max_rel
            best_max_rel = max_rel
            best_idx = k
        end
        @printf("  h=%.1e: max=%.3e mean=%.3e l2=%.3e (FD %.2fs)\n",
                h, max_rel, mean_rel, l2_rel, t_fd)
        flush(stdout)
    end

    outpath = joinpath(DATADIR, "gradient_fd_stepsweep.csv")
    open(outpath, "w") do io
        write(io, "# Dispersive gradient FD step-size sweep (AD reference fixed)\n")
        write(io, "h,fd_time_s,max_rel_error,mean_rel_error,l2_rel_error\n")
        for row in rows
            @printf(io, "%.9e,%.6f,%.12e,%.12e,%.12e\n",
                    row.h, row.fd_time_s, row.max_rel, row.mean_rel, row.l2_rel)
        end
        best = rows[best_idx]
        @printf(io, "# best_h=%.9e,best_max_rel=%.12e,best_l2_rel=%.12e\n",
                best.h, best.max_rel, best.l2_rel)
    end
    println("  Saved: $outpath")
end

function small_mask_around_pipe(ctx; half_w::Int=3, half_h::Int=3)
    mask = falses(ctx.nx, ctx.ny)
    for j in (ctx.pipe_cy-half_h):(ctx.pipe_cy+half_h), i in (ctx.pipe_cx-half_w):(ctx.pipe_cx+half_w)
        if 1 <= i <= ctx.nx && 1 <= j <= ctx.ny
            mask[i, j] = true
        end
    end
    return mask
end

function mask_metrics(ctx, mask::BitMatrix, eps_est::Matrix{Float64})
    truth = Float64[]
    est = Float64[]
    for j in 1:ctx.ny, i in 1:ctx.nx
        if mask[i, j]
            push!(truth, ctx.eps_inf_true[i, j])
            push!(est, eps_est[i, j])
        end
    end
    rmse = sqrt(mean((est .- truth).^2))
    peak_true = maximum(truth)
    peak_est = maximum(est)
    recovery = 100.0 * peak_est / peak_true
    return rmse, peak_true, peak_est, recovery
end

function multi_source_gradient_ad(x, ctx, mask, eps_init, deps_init, tau_init, sigma_init)
    g = zeros(length(x))
    for k in eachindex(ctx.configs)
        dx_k = zeros(length(x))
        Enzyme.autodiff(Enzyme.Reverse, forward_misfit, Enzyme.Active,
                        Enzyme.Duplicated(x, dx_k),
                        Enzyme.Const(ctx.configs[k]),
                        Enzyme.Const(ctx.obs_datas_clean[k]),
                        Enzyme.Const(ctx.src_waveforms[k]),
                        Enzyme.Const(eps_init),
                        Enzyme.Const(deps_init),
                        Enzyme.Const(tau_init),
                        Enzyme.Const(sigma_init),
                        Enzyme.Const(mask),
                        Enzyme.Const(:eps_inf))
        g .+= dx_k
    end
    return g
end

function multi_source_objective(x, ctx, mask, eps_init, deps_init, tau_init, sigma_init)
    total = 0.0
    for k in eachindex(ctx.configs)
        total += forward_misfit(
            x, ctx.configs[k], ctx.obs_datas_clean[k], ctx.src_waveforms[k],
            eps_init, deps_init, tau_init, sigma_init, mask, :eps_inf,
        )
    end
    return total
end

function run_full_physics_comparator()
    println("Running R3: full-physics AD-vs-FD comparator...")
    ctx = build_reduced_multisource_context()
    eps_init, deps_init, tau_init, sigma_init = build_initial_model(ctx)
    mask = small_mask_around_pipe(ctx; half_w=3, half_h=3)
    n_params = count(mask)
    max_iter = 6

    @printf("  Domain: %d x %d, sources=%d, n_params=%d\n",
            ctx.nx, ctx.ny, length(ctx.configs), n_params)
    flush(stdout)

    x0 = Float64[]
    for j in 1:ctx.ny, i in 1:ctx.nx
        if mask[i, j]
            push!(x0, eps_init[i, j])
        end
    end

    obj = x -> multi_source_objective(x, ctx, mask, eps_init, deps_init, tau_init, sigma_init)

    # Warm-up (exclude first-call compilation from comparator timing).
    multi_source_gradient_ad(x0, ctx, mask, eps_init, deps_init, tau_init, sigma_init)
    fd_gradient(obj, x0; h=1e-5)

    t_grad_ad = @elapsed g_ad = multi_source_gradient_ad(x0, ctx, mask, eps_init, deps_init, tau_init, sigma_init)
    t_grad_fd = @elapsed g_fd = fd_gradient(obj, x0; h=1e-5)

    rel = abs.(g_ad .- g_fd) ./ max.(abs.(g_fd), 1e-30)
    grad_max_rel = maximum(rel)
    grad_mean_rel = mean(rel)
    grad_l2_rel = norm(g_ad - g_fd) / max(norm(g_fd), 1e-30)
    @printf("  Initial gradient AD-vs-FD: max=%.3e mean=%.3e l2=%.3e\n",
            grad_max_rel, grad_mean_rel, grad_l2_rel)
    flush(stdout)

    # Warm-up optimization wrappers
    run_fwi_multisource(
        ctx.configs, ctx.obs_datas_clean, ctx.src_waveforms,
        eps_init, deps_init, tau_init, sigma_init, mask;
        max_iter=1, param_type=:eps_inf, use_ad=true, verbose=false,
        lower_bound=1.0, upper_bound=25.0, lambda=1.0,
    )
    run_fwi_multisource(
        ctx.configs, ctx.obs_datas_clean, ctx.src_waveforms,
        eps_init, deps_init, tau_init, sigma_init, mask;
        max_iter=1, param_type=:eps_inf, use_ad=false, verbose=false,
        lower_bound=1.0, upper_bound=25.0, lambda=1.0,
    )

    t_ad = @elapsed result_ad = run_fwi_multisource(
        ctx.configs, ctx.obs_datas_clean, ctx.src_waveforms,
        eps_init, deps_init, tau_init, sigma_init, mask;
        max_iter=max_iter, param_type=:eps_inf, use_ad=true, verbose=true,
        lower_bound=1.0, upper_bound=25.0, lambda=1.0,
    )

    t_fd = @elapsed result_fd = run_fwi_multisource(
        ctx.configs, ctx.obs_datas_clean, ctx.src_waveforms,
        eps_init, deps_init, tau_init, sigma_init, mask;
        max_iter=max_iter, param_type=:eps_inf, use_ad=false, verbose=true,
        lower_bound=1.0, upper_bound=25.0, lambda=1.0,
    )

    rmse_ad, peak_true, peak_est_ad, rec_ad = mask_metrics(ctx, mask, result_ad.eps_inf_est)
    rmse_fd, _, peak_est_fd, rec_fd = mask_metrics(ctx, mask, result_fd.eps_inf_est)

    reduction_ad = 100.0 * (1.0 - result_ad.loss_history[end] / result_ad.loss_history[1])
    reduction_fd = 100.0 * (1.0 - result_fd.loss_history[end] / result_fd.loss_history[1])

    conv_ad = joinpath(DATADIR, "full_physics_comparator_ad_convergence.csv")
    conv_fd = joinpath(DATADIR, "full_physics_comparator_fd_convergence.csv")
    save_convergence_csv(conv_ad, result_ad;
        header="Full-physics comparator AD run, n_params=$n_params, max_iter=$max_iter")
    save_convergence_csv(conv_fd, result_fd;
        header="Full-physics comparator FD run, n_params=$n_params, max_iter=$max_iter")

    summary = joinpath(DATADIR, "full_physics_comparator_summary.csv")
    open(summary, "w") do io
        write(io, "# Full-physics Debye+CPML comparator with matched iteration budget\n")
        write(io, "# Gradient timings are steady-state (after warm-up)\n")
        write(io, "method,n_params,max_iter,gradient_time_s,initial_grad_max_rel_error,initial_grad_mean_rel_error,initial_grad_l2_rel_error,total_runtime_s,loss_initial,loss_final,loss_reduction_pct,rmse,peak_true,peak_est,peak_recovery_pct\n")
        @printf(io, "AD,%d,%d,%.6f,%.12e,%.12e,%.12e,%.6f,%.12e,%.12e,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                n_params, max_iter, t_grad_ad, grad_max_rel, grad_mean_rel, grad_l2_rel,
                t_ad, result_ad.loss_history[1], result_ad.loss_history[end], reduction_ad,
                rmse_ad, peak_true, peak_est_ad, rec_ad)
        @printf(io, "FD,%d,%d,%.6f,%.12e,%.12e,%.12e,%.6f,%.12e,%.12e,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                n_params, max_iter, t_grad_fd, grad_max_rel, grad_mean_rel, grad_l2_rel,
                t_fd, result_fd.loss_history[1], result_fd.loss_history[end], reduction_fd,
                rmse_fd, peak_true, peak_est_fd, rec_fd)
    end

    @printf("  AD runtime %.1fs, FD runtime %.1fs\n", t_ad, t_fd)
    @printf("  AD reduction %.2f%% (RMSE %.3f), FD reduction %.2f%% (RMSE %.3f)\n",
            reduction_ad, rmse_ad, reduction_fd, rmse_fd)
    println("  Saved: $conv_ad")
    println("  Saved: $conv_fd")
    println("  Saved: $summary")
end

function main()
    Random.seed!(20260312)
    run_gradient_stepsweep()
    run_full_physics_comparator()
    println("\n=== Revision validation artifacts complete ===")
end

main()
