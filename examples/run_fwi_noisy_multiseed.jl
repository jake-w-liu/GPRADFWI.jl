# run_fwi_noisy_multiseed.jl
# Multi-seed noise robustness statistics for revision item M2.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using DelimitedFiles
using Printf
using Random
using Statistics
using GPRADFWI

include(joinpath(@__DIR__, "revision_reduced_common.jl"))

const DATADIR = joinpath(@__DIR__, "..", "..", "paper", "data")
mkpath(DATADIR)

function add_noise_to_data(clean_data::Matrix{Float64}, snr_db::Int)
    signal_power = sum(clean_data .^ 2)
    noise_power = signal_power / (10.0 ^ (snr_db / 10.0))
    noise_std = sqrt(noise_power / length(clean_data))
    noise = noise_std .* randn(size(clean_data))
    noisy = clean_data .+ noise
    actual_snr = 10.0 * log10(signal_power / sum(noise .^ 2))
    return noisy, actual_snr, noise_std
end

function centerline_profile(ctx, eps_est::Matrix{Float64})
    prof = Float64[]
    for j in ctx.inv_y_lo:ctx.inv_y_hi
        push!(prof, eps_est[ctx.pipe_cx, j])
    end
    return prof
end

function main()
    Random.seed!(20260303)

    println("Building reduced-domain context for multi-seed noise study...")
    ctx = build_reduced_multisource_context()
    @printf("  Domain: %d x %d, nt=%d\n", ctx.nx, ctx.ny, ctx.configs[1].nt)
    @printf("  Sources: %d, parameters: %d\n", length(ctx.configs), ctx.n_params)
    flush(stdout)

    src_sel = [1, 2]  # speed-oriented Monte Carlo setup
    configs = ctx.configs[src_sel]
    src_waveforms = ctx.src_waveforms[src_sel]
    obs_clean = ctx.obs_datas_clean[src_sel]

    eps_init, deps_init, tau_init, sigma_init = build_initial_model(ctx)

    max_iter = 50
    snr_levels = [40, 30, 20]
    n_seeds = 10

    println("Running clean reference inversion...")
    clean_result = run_fwi_multisource(
        configs,
        obs_clean,
        src_waveforms,
        eps_init,
        deps_init,
        tau_init,
        sigma_init,
        ctx.param_mask;
        max_iter=max_iter,
        param_type=:eps_inf,
        use_ad=true,
        verbose=true,
        lower_bound=1.0,
        upper_bound=25.0,
        lambda=1.0,
    )
    clean_rmse, clean_peak_true, clean_peak_est, clean_recovery =
        compute_region_metrics(ctx, clean_result.eps_inf_est)
    clean_reduction = 100.0 * (1.0 - clean_result.loss_history[end] / clean_result.loss_history[1])
    clean_profile = centerline_profile(ctx, clean_result.eps_inf_est)

    clean_conv_file = joinpath(DATADIR, "fwi_noisy_multiseed_clean_convergence.csv")
    save_convergence_csv(clean_conv_file, clean_result;
        header="Reduced-domain clean reference, max_iter=$(max_iter)")

    records = NamedTuple[]
    summary_rows = NamedTuple[]
    convergence_rows = NamedTuple[]

    ndepth = ctx.inv_y_hi - ctx.inv_y_lo + 1

    for snr_db in snr_levels
        println("\n" * "="^70)
        @printf("SNR = %d dB (%d seeds)\n", snr_db, n_seeds)
        println("="^70)
        flush(stdout)

        loss_mat = zeros(n_seeds, max_iter + 1)
        profile_mat = zeros(n_seeds, ndepth)
        rmse_vec = zeros(n_seeds)
        peak_vec = zeros(n_seeds)
        recov_vec = zeros(n_seeds)
        red_vec = zeros(n_seeds)
        runtime_vec = zeros(n_seeds)
        snr_actual_vec = zeros(n_seeds)

        for s in 1:n_seeds
            seed_id = 10000 + 100 * snr_db + s
            Random.seed!(seed_id)

            obs_noisy = Matrix{Float64}[]
            actual_snr_acc = 0.0
            for k in eachindex(obs_clean)
                dn, actual_snr, _ = add_noise_to_data(obs_clean[k], snr_db)
                push!(obs_noisy, dn)
                actual_snr_acc += actual_snr
            end
            snr_actual = actual_snr_acc / length(obs_clean)

            t0 = time()
            result = run_fwi_multisource(
                configs,
                obs_noisy,
                src_waveforms,
                eps_init,
                deps_init,
                tau_init,
                sigma_init,
                ctx.param_mask;
                max_iter=max_iter,
                param_type=:eps_inf,
                use_ad=true,
                verbose=false,
                lower_bound=1.0,
                upper_bound=25.0,
                lambda=1.0,
            )
            runtime_s = time() - t0

            rmse, peak_true, peak_est, recovery = compute_region_metrics(ctx, result.eps_inf_est)
            reduction = 100.0 * (1.0 - result.loss_history[end] / result.loss_history[1])

            loss_mat[s, :] .= result.loss_history
            profile_mat[s, :] .= centerline_profile(ctx, result.eps_inf_est)
            rmse_vec[s] = rmse
            peak_vec[s] = peak_est
            recov_vec[s] = recovery
            red_vec[s] = reduction
            runtime_vec[s] = runtime_s
            snr_actual_vec[s] = snr_actual

            push!(records, (
                snr_db=snr_db,
                seed_id=seed_id,
                snr_actual_db=snr_actual,
                runtime_s=runtime_s,
                loss_initial=result.loss_history[1],
                loss_final=result.loss_history[end],
                loss_reduction_pct=reduction,
                rmse=rmse,
                peak_true=peak_true,
                peak_est=peak_est,
                peak_recovery_pct=recovery,
            ))

            @printf("  seed %d/%d: actual SNR %.2f dB, RMSE %.3f, recovery %.1f%%\n",
                    s, n_seeds, snr_actual, rmse, recovery)
            flush(stdout)
        end

        push!(summary_rows, (
            snr_db=snr_db,
            n_seeds=n_seeds,
            snr_actual_mean=mean(snr_actual_vec),
            snr_actual_std=std(snr_actual_vec),
            runtime_mean=mean(runtime_vec),
            runtime_std=std(runtime_vec),
            rmse_mean=mean(rmse_vec),
            rmse_std=std(rmse_vec),
            peak_mean=mean(peak_vec),
            peak_std=std(peak_vec),
            recovery_mean=mean(recov_vec),
            recovery_std=std(recov_vec),
            reduction_mean=mean(red_vec),
            reduction_std=std(red_vec),
            clean_peak_true=clean_peak_true,
        ))

        for iter in 0:max_iter
            push!(convergence_rows, (
                snr_db=snr_db,
                iteration=iter,
                loss_mean=mean(loss_mat[:, iter+1]),
                loss_std=std(loss_mat[:, iter+1]),
            ))
        end

        recon_stat_file = joinpath(DATADIR, "fwi_noisy_multiseed_reconstruction_stats_snr$(snr_db)db.csv")
        open(recon_stat_file, "w") do io
            write(io, "# Multi-seed reconstruction stats, SNR=$(snr_db)dB, n_seeds=$(n_seeds)\n")
            write(io, "depth_cm,eps_true,eps_initial,eps_clean,eps_mean,eps_std\n")
            for (idx, j) in enumerate(ctx.inv_y_lo:ctx.inv_y_hi)
                depth_cm = (j - ctx.surface_j) * ctx.grid_dx * 100.0
                eps_true = ctx.eps_inf_true[ctx.pipe_cx, j]
                eps0 = eps_init[ctx.pipe_cx, j]
                eps_clean = clean_profile[idx]
                eps_mean = mean(profile_mat[:, idx])
                eps_std = std(profile_mat[:, idx])
                @printf(io, "%.2f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                        depth_cm, eps_true, eps0, eps_clean, eps_mean, eps_std)
            end
        end
        println("  Saved: $recon_stat_file")
    end

    records_file = joinpath(DATADIR, "fwi_noisy_multiseed_records.csv")
    open(records_file, "w") do io
        write(io, "# Multi-seed noisy FWI records\n")
        write(io, "snr_db,seed_id,snr_actual_db,runtime_s,loss_initial,loss_final,loss_reduction_pct,rmse,peak_true,peak_est,peak_recovery_pct\n")
        for r in records
            @printf(io, "%d,%d,%.6f,%.6f,%.12e,%.12e,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    r.snr_db, r.seed_id, r.snr_actual_db, r.runtime_s,
                    r.loss_initial, r.loss_final, r.loss_reduction_pct,
                    r.rmse, r.peak_true, r.peak_est, r.peak_recovery_pct)
        end
    end

    summary_file = joinpath(DATADIR, "fwi_noisy_multiseed_summary.csv")
    open(summary_file, "w") do io
        write(io, "# Multi-seed noisy FWI summary by SNR\n")
        write(io, "snr_db,n_seeds,snr_actual_mean,snr_actual_std,runtime_mean,runtime_std,rmse_mean,rmse_std,peak_mean,peak_std,recovery_mean,recovery_std,reduction_mean,reduction_std,clean_peak_true\n")
        for r in summary_rows
            @printf(io, "%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    r.snr_db, r.n_seeds,
                    r.snr_actual_mean, r.snr_actual_std,
                    r.runtime_mean, r.runtime_std,
                    r.rmse_mean, r.rmse_std,
                    r.peak_mean, r.peak_std,
                    r.recovery_mean, r.recovery_std,
                    r.reduction_mean, r.reduction_std,
                    r.clean_peak_true)
        end
        @printf(io, "clean_reference,%d,NaN,NaN,NaN,NaN,%.6f,NaN,%.6f,NaN,%.6f,NaN,%.6f,NaN,%.6f\n",
                1, clean_rmse, clean_peak_est, clean_recovery, clean_reduction, clean_peak_true)
    end

    conv_stats_file = joinpath(DATADIR, "fwi_noisy_multiseed_convergence_stats.csv")
    open(conv_stats_file, "w") do io
        write(io, "# Multi-seed convergence statistics (mean/std)\n")
        write(io, "snr_db,iteration,loss_mean,loss_std\n")
        for r in convergence_rows
            @printf(io, "%d,%d,%.12e,%.12e\n", r.snr_db, r.iteration, r.loss_mean, r.loss_std)
        end
    end

    println("\nSaved: $clean_conv_file")
    println("Saved: $records_file")
    println("Saved: $summary_file")
    println("Saved: $conv_stats_file")
    println("\n=== Multi-seed noise study complete ===")
end

main()
