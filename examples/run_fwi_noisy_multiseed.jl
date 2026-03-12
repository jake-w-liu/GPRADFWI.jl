# run_fwi_noisy_multiseed.jl
# Multi-seed noise robustness study with strict comparability to Fig. 7 setup.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using GPRADFWI
using Printf
using Random
using Statistics

const DATADIR = joinpath(@__DIR__, "..", "..", "paper", "data")
mkpath(DATADIR)

function add_noise_to_data(clean_data::Matrix{Float64}, snr_db::Int)
    signal_power = sum(clean_data .^ 2)
    noise_power = signal_power / (10.0 ^ (snr_db / 10.0))
    noise_std = sqrt(noise_power / length(clean_data))
    noise = noise_std .* randn(size(clean_data))
    noisy = clean_data .+ noise
    actual_snr = 10.0 * log10(signal_power / sum(noise .^ 2))
    return noisy, actual_snr
end

function build_full_domain_context()
    domain_x = 1.0
    domain_y = 0.85
    grid_dx = 0.005
    fc_gpr = 500e6

    nx = round(Int, domain_x / grid_dx)
    ny = round(Int, domain_y / grid_dx)
    npml = 10

    rx_y = npml + 10
    rx_x_list = collect((npml + 3):4:(nx - npml - 3))

    src_x_list = [30, 65, 100, 135, 170]
    nsrc = length(src_x_list)

    configs = FDTDConfig[]
    src_waveforms = Vector{Float64}[]
    for sx in src_x_list
        cfg = create_config(
            nx=nx, ny=ny, dx=grid_dx, fc=fc_gpr, npml=npml,
            src_ix=sx, src_iy=rx_y,
            rx_iy=rx_y, rx_ix_list=rx_x_list,
        )
        push!(configs, cfg)
        push!(src_waveforms, create_source(cfg))
    end

    eps_inf_true = ones(nx, ny)
    deps_true = zeros(nx, ny)
    tau_true = zeros(nx, ny)
    sigma_true = zeros(nx, ny)

    surface_j = npml + 15
    layer1_top = surface_j
    layer1_bot = surface_j + round(Int, 0.6 / grid_dx)

    for j in layer1_top:min(layer1_bot, ny), i in 1:nx
        eps_inf_true[i, j] = 4.0
        deps_true[i, j] = 4.0
        tau_true[i, j] = 0.3e-9
        sigma_true[i, j] = 0.005
    end

    layer2_top = layer1_bot + 1
    for j in layer2_top:ny, i in 1:nx
        eps_inf_true[i, j] = 6.0
        deps_true[i, j] = 12.0
        tau_true[i, j] = 1.0e-9
        sigma_true[i, j] = 0.02
    end

    pipe_cx = nx ÷ 2
    pipe_cy = surface_j + round(Int, 0.4 / grid_dx)
    pipe_r = round(Int, 0.05 / grid_dx)

    for j in 1:ny, i in 1:nx
        if (i - pipe_cx)^2 + (j - pipe_cy)^2 <= pipe_r^2
            eps_inf_true[i, j] = 15.0
            deps_true[i, j] = 10.0
            tau_true[i, j] = 0.5e-9
            sigma_true[i, j] = 0.001
        end
    end

    inv_x_lo = pipe_cx - 25
    inv_x_hi = pipe_cx + 25
    inv_y_lo = pipe_cy - 20
    inv_y_hi = pipe_cy + 20

    param_mask = falses(nx, ny)
    for j in inv_y_lo:inv_y_hi, i in inv_x_lo:inv_x_hi
        param_mask[i, j] = true
    end

    obs_datas_clean = Matrix{Float64}[]
    for k in 1:nsrc
        od = run_forward!(configs[k], eps_inf_true, deps_true, tau_true, sigma_true, src_waveforms[k])
        push!(obs_datas_clean, od)
    end

    return (
        nx=nx,
        ny=ny,
        grid_dx=grid_dx,
        configs=configs,
        src_waveforms=src_waveforms,
        eps_inf_true=eps_inf_true,
        deps_true=deps_true,
        tau_true=tau_true,
        sigma_true=sigma_true,
        surface_j=surface_j,
        layer1_top=layer1_top,
        layer1_bot=layer1_bot,
        layer2_top=layer2_top,
        pipe_cx=pipe_cx,
        inv_x_lo=inv_x_lo,
        inv_x_hi=inv_x_hi,
        inv_y_lo=inv_y_lo,
        inv_y_hi=inv_y_hi,
        param_mask=param_mask,
        n_params=count(param_mask),
        obs_datas_clean=obs_datas_clean,
    )
end

function build_initial_model(ctx)
    nx, ny = ctx.nx, ctx.ny

    eps_inf = ones(nx, ny)
    deps = zeros(nx, ny)
    tau = zeros(nx, ny)
    sigma = zeros(nx, ny)

    for j in ctx.layer1_top:min(ctx.layer1_bot, ny), i in 1:nx
        eps_inf[i, j] = 4.0
        deps[i, j] = 4.0
        tau[i, j] = 0.3e-9
        sigma[i, j] = 0.005
    end
    for j in ctx.layer2_top:ny, i in 1:nx
        eps_inf[i, j] = 6.0
        deps[i, j] = 12.0
        tau[i, j] = 1.0e-9
        sigma[i, j] = 0.02
    end

    return eps_inf, deps, tau, sigma
end

function centerline_profile(ctx, eps_est::Matrix{Float64})
    profile = Float64[]
    for j in ctx.inv_y_lo:ctx.inv_y_hi
        push!(profile, eps_est[ctx.pipe_cx, j])
    end
    return profile
end

function compute_region_metrics(ctx, eps_est::Matrix{Float64})
    eps_true_region = Float64[]
    eps_est_region = Float64[]
    for j in ctx.inv_y_lo:ctx.inv_y_hi, i in ctx.inv_x_lo:ctx.inv_x_hi
        push!(eps_true_region, ctx.eps_inf_true[i, j])
        push!(eps_est_region, eps_est[i, j])
    end

    rmse = sqrt(mean((eps_true_region .- eps_est_region) .^ 2))
    peak_true = maximum(ctx.eps_inf_true[ctx.inv_x_lo:ctx.inv_x_hi, ctx.inv_y_lo:ctx.inv_y_hi])
    peak_est = maximum(eps_est[ctx.inv_x_lo:ctx.inv_x_hi, ctx.inv_y_lo:ctx.inv_y_hi])
    recovery = 100.0 * peak_est / peak_true
    return rmse, peak_true, peak_est, recovery
end

function save_convergence_csv(path::String, result; header::String="")
    open(path, "w") do io
        if !isempty(header)
            write(io, "# $header\n")
        end
        write(io, "iteration,loss,grad_norm\n")
        for k in eachindex(result.loss_history)
            @printf(io, "%d,%.12e,%.12e\n", k - 1, result.loss_history[k], result.grad_norm_history[k])
        end
    end
end

function main()
    Random.seed!(20260303)

    println("Building full-domain context (identical to Fig. 7 setup)...")
    ctx = build_full_domain_context()
    @printf("  Domain: %d x %d\n", ctx.nx, ctx.ny)
    @printf("  Sources: %d, parameters: %d\n", length(ctx.configs), ctx.n_params)
    flush(stdout)

    eps_init, deps_init, tau_init, sigma_init = build_initial_model(ctx)

    max_iter = 50
    snr_levels = [40, 30, 20]
    n_seeds = 10

    println("Running clean reference inversion...")
    clean_result = run_fwi_multisource(
        ctx.configs,
        ctx.obs_datas_clean,
        ctx.src_waveforms,
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
        header="Full-domain clean reference (Fig. 7 setup), max_iter=$(max_iter)")

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
            for k in eachindex(ctx.obs_datas_clean)
                dn, actual_snr = add_noise_to_data(ctx.obs_datas_clean[k], snr_db)
                push!(obs_noisy, dn)
                actual_snr_acc += actual_snr
            end
            snr_actual = actual_snr_acc / length(ctx.obs_datas_clean)

            t0 = time()
            result = run_fwi_multisource(
                ctx.configs,
                obs_noisy,
                ctx.src_waveforms,
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
                loss_mean=mean(loss_mat[:, iter + 1]),
                loss_std=std(loss_mat[:, iter + 1]),
            ))
        end

        recon_stat_file = joinpath(DATADIR, "fwi_noisy_multiseed_reconstruction_stats_snr$(snr_db)db.csv")
        open(recon_stat_file, "w") do io
            write(io, "# Multi-seed reconstruction stats, full-domain setup, SNR=$(snr_db)dB, n_seeds=$(n_seeds)\n")
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
        write(io, "# Multi-seed noisy FWI records (full-domain setup)\n")
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
        write(io, "# Multi-seed noisy FWI summary by SNR (full-domain setup)\n")
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
        write(io, "# Multi-seed convergence statistics (mean/std, full-domain setup)\n")
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
