# run_fwi_joint_mitigation.jl
# Reduced-domain joint inversion mitigation experiment for epsilon_inf-sigma cross-talk.
#
# Produces:
#   paper/data/fwi_joint_mitigation_summary.csv
#   paper/data/fwi_joint_mitigation_baseline_convergence.csv
#   paper/data/fwi_joint_mitigation_damped_convergence.csv
#   paper/data/fwi_joint_mitigation_reconstruction_1d.csv

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using GPRADFWI
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "revision_reduced_common.jl"))

const DATADIR = joinpath(@__DIR__, "..", "..", "paper", "data")
mkpath(DATADIR)

function compute_joint_metrics(ctx, result)
    eps_true = Float64[]
    eps_est = Float64[]
    sigma_true = Float64[]
    sigma_est = Float64[]

    sigma_bound_hits = 0
    n_cells = 0

    pipe_sigma_true = Float64[]
    pipe_sigma_est = Float64[]
    bg_sigma_true = Float64[]
    bg_sigma_est = Float64[]

    for j in ctx.inv_y_lo:ctx.inv_y_hi, i in ctx.inv_x_lo:ctx.inv_x_hi
        et = ctx.eps_inf_true[i, j]
        ee = result.eps_inf_est[i, j]
        st = ctx.sigma_true[i, j]
        se = result.sigma_est[i, j]

        push!(eps_true, et)
        push!(eps_est, ee)
        push!(sigma_true, st)
        push!(sigma_est, se)

        if se <= 1e-6 || se >= 0.099999
            sigma_bound_hits += 1
        end
        n_cells += 1

        if (i - ctx.pipe_cx)^2 + (j - ctx.pipe_cy)^2 <= ctx.pipe_r^2
            push!(pipe_sigma_true, st)
            push!(pipe_sigma_est, se)
        else
            push!(bg_sigma_true, st)
            push!(bg_sigma_est, se)
        end
    end

    loss0 = result.loss_history[1]
    lossf = result.loss_history[end]
    loss_reduction = 100.0 * (1.0 - lossf / loss0)

    eps_rmse = sqrt(mean((eps_est .- eps_true).^2))
    eps_peak_true = maximum(eps_true)
    eps_peak_est = maximum(eps_est)
    eps_peak_recovery = 100.0 * eps_peak_est / eps_peak_true

    sigma_rmse = sqrt(mean((sigma_est .- sigma_true).^2))
    sigma_bound_frac = sigma_bound_hits / n_cells

    sigma_pipe_true = mean(pipe_sigma_true)
    sigma_bg_true = mean(bg_sigma_true)
    sigma_pipe_est = mean(pipe_sigma_est)
    sigma_bg_est = mean(bg_sigma_est)
    sigma_contrast_true = sigma_pipe_true - sigma_bg_true
    sigma_contrast_est = sigma_pipe_est - sigma_bg_est

    return (
        loss0=loss0,
        lossf=lossf,
        loss_reduction=loss_reduction,
        eps_rmse=eps_rmse,
        eps_peak_true=eps_peak_true,
        eps_peak_est=eps_peak_est,
        eps_peak_recovery=eps_peak_recovery,
        sigma_rmse=sigma_rmse,
        sigma_bound_frac=sigma_bound_frac,
        sigma_pipe_true=sigma_pipe_true,
        sigma_bg_true=sigma_bg_true,
        sigma_pipe_est=sigma_pipe_est,
        sigma_bg_est=sigma_bg_est,
        sigma_contrast_true=sigma_contrast_true,
        sigma_contrast_est=sigma_contrast_est,
    )
end

function run_joint_case(ctx; max_iter::Int, lambda_sigma::Float64, lambda_damp_sigma::Float64, label::String)
    eps_init, deps_init, tau_init, sigma_init = build_initial_model(ctx)

    @printf("\n--- %s ---\n", label)
    @printf("max_iter=%d, lambda_sigma=%.2e, lambda_damp_sigma=%.2e\n",
            max_iter, lambda_sigma, lambda_damp_sigma)
    flush(stdout)

    result = run_fwi_multisource(
        ctx.configs, ctx.obs_datas_clean, ctx.src_waveforms,
        eps_init, deps_init, tau_init, sigma_init, ctx.param_mask;
        max_iter=max_iter, param_type=:both, use_ad=true, verbose=true,
        lower_bound=1.0, upper_bound=25.0,
        lower_bound_sigma=0.0, upper_bound_sigma=0.1,
        lambda=1.0, lambda_sigma=lambda_sigma,
        lambda_damp=0.0, lambda_damp_sigma=lambda_damp_sigma,
    )

    metrics = compute_joint_metrics(ctx, result)
    @printf("  loss reduction = %.2f%%\n", metrics.loss_reduction)
    @printf("  eps RMSE = %.3f, peak recovery = %.2f%%\n",
            metrics.eps_rmse, metrics.eps_peak_recovery)
    @printf("  sigma RMSE = %.5f S/m, bound-hit frac = %.2f%%\n",
            metrics.sigma_rmse, 100.0 * metrics.sigma_bound_frac)
    @printf("  sigma contrast true(pipe-bg)=%.5f, estimated=%.5f [S/m]\n",
            metrics.sigma_contrast_true, metrics.sigma_contrast_est)
    flush(stdout)

    return result, metrics
end

function save_joint_1d_comparison(path::String, ctx, eps_init::Matrix{Float64}, sigma_init::Matrix{Float64},
                                  result_baseline, result_damped)
    open(path, "w") do io
        write(io, "# Reduced-domain joint inversion mitigation comparison (vertical slice through pipe center)\n")
        write(io, "depth_cm,eps_true,eps_init,eps_baseline,eps_damped,sigma_true,sigma_init,sigma_baseline,sigma_damped\n")
        for j in ctx.inv_y_lo:ctx.inv_y_hi
            depth_cm = (j - ctx.surface_j) * ctx.grid_dx * 100.0
            @printf(io, "%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    depth_cm,
                    ctx.eps_inf_true[ctx.pipe_cx, j],
                    eps_init[ctx.pipe_cx, j],
                    result_baseline.eps_inf_est[ctx.pipe_cx, j],
                    result_damped.eps_inf_est[ctx.pipe_cx, j],
                    ctx.sigma_true[ctx.pipe_cx, j],
                    sigma_init[ctx.pipe_cx, j],
                    result_baseline.sigma_est[ctx.pipe_cx, j],
                    result_damped.sigma_est[ctx.pipe_cx, j])
        end
    end
end

function save_summary(path::String, max_iter::Int, lambda_sigma::Float64, lambda_damp_sigma::Float64,
                      baseline_metrics, damped_metrics)
    open(path, "w") do io
        write(io, "# Reduced-domain joint inversion mitigation summary\n")
        write(io, "# setup: 80x64, 3 sources, Debye+CPML, joint epsilon_inf-sigma, AD gradients\n")
        write(io, "method,max_iter,lambda_sigma,lambda_damp_sigma,loss_initial,loss_final,loss_reduction_pct,eps_rmse,eps_peak_true,eps_peak_est,eps_peak_recovery_pct,sigma_rmse,sigma_bound_fraction,sigma_pipe_mean,sigma_bg_mean,sigma_contrast_pipe_minus_bg\n")
        @printf(io, "baseline,%d,%.6e,%.6e,%.12e,%.12e,%.6f,%.6f,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                max_iter, lambda_sigma, 0.0,
                baseline_metrics.loss0, baseline_metrics.lossf, baseline_metrics.loss_reduction,
                baseline_metrics.eps_rmse, baseline_metrics.eps_peak_true, baseline_metrics.eps_peak_est,
                baseline_metrics.eps_peak_recovery, baseline_metrics.sigma_rmse,
                baseline_metrics.sigma_bound_frac, baseline_metrics.sigma_pipe_est,
                baseline_metrics.sigma_bg_est, baseline_metrics.sigma_contrast_est)
        @printf(io, "sigma_damped,%d,%.6e,%.6e,%.12e,%.12e,%.6f,%.6f,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                max_iter, lambda_sigma, lambda_damp_sigma,
                damped_metrics.loss0, damped_metrics.lossf, damped_metrics.loss_reduction,
                damped_metrics.eps_rmse, damped_metrics.eps_peak_true, damped_metrics.eps_peak_est,
                damped_metrics.eps_peak_recovery, damped_metrics.sigma_rmse,
                damped_metrics.sigma_bound_frac, damped_metrics.sigma_pipe_est,
                damped_metrics.sigma_bg_est, damped_metrics.sigma_contrast_est)
    end
end

function main()
    Random.seed!(20260312)

    ctx = build_reduced_multisource_context()
    eps_init, _, _, sigma_init = build_initial_model(ctx)

    max_iter = 30
    lambda_sigma = 1e4
    lambda_damp_sigma = 5e3

    baseline_result, baseline_metrics = run_joint_case(
        ctx;
        max_iter=max_iter,
        lambda_sigma=lambda_sigma,
        lambda_damp_sigma=0.0,
        label="Baseline joint inversion (no sigma damping)",
    )

    damped_result, damped_metrics = run_joint_case(
        ctx;
        max_iter=max_iter,
        lambda_sigma=lambda_sigma,
        lambda_damp_sigma=lambda_damp_sigma,
        label="Mitigated joint inversion (sigma norm damping)",
    )

    conv_baseline = joinpath(DATADIR, "fwi_joint_mitigation_baseline_convergence.csv")
    conv_damped = joinpath(DATADIR, "fwi_joint_mitigation_damped_convergence.csv")
    summary_csv = joinpath(DATADIR, "fwi_joint_mitigation_summary.csv")
    recon_1d_csv = joinpath(DATADIR, "fwi_joint_mitigation_reconstruction_1d.csv")

    save_convergence_csv(conv_baseline, baseline_result;
        header="Reduced-domain joint baseline (no sigma damping)")
    save_convergence_csv(conv_damped, damped_result;
        header="Reduced-domain joint mitigated (sigma damping)")
    save_summary(summary_csv, max_iter, lambda_sigma, lambda_damp_sigma,
                 baseline_metrics, damped_metrics)
    save_joint_1d_comparison(recon_1d_csv, ctx, eps_init, sigma_init,
                             baseline_result, damped_result)

    @printf("\nSaved:\n  %s\n  %s\n  %s\n  %s\n",
            conv_baseline, conv_damped, summary_csv, recon_1d_csv)
    println("\n=== Joint mitigation experiment complete ===")
end

main()
