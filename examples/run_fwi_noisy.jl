# run_fwi_noisy.jl — Noise robustness study (M1)
#
# Adds Gaussian noise at 3 SNR levels (20, 30, 40 dB) to the observed data
# from the dispersive true model, then runs multi-source FWI for each.
#
# SNR is defined as: SNR_dB = 10 * log10(||d_obs||² / ||noise||²)
# so noise_std = ||d_obs|| / (10^(SNR_dB/20) * sqrt(N))
#
# Outputs to paper/data/:
#   fwi_noisy_snrXXdb_convergence.csv
#   fwi_noisy_snrXXdb_reconstruction_1d.csv
#   fwi_noisy_summary.csv

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using GPRADFWI
using DelimitedFiles
using Printf
using Random
using LinearAlgebra
using Statistics

datadir = joinpath(@__DIR__, "..", "..", "paper", "data")
mkpath(datadir)

# ══════════════════════════════════════════════════════════════════════
# Domain setup — identical to run_fwi_large_domain.jl
# ══════════════════════════════════════════════════════════════════════
const domain_x = 1.0
const domain_y = 0.85
const grid_dx  = 0.005
const fc_gpr   = 500e6

const nx = round(Int, domain_x / grid_dx)
const ny = round(Int, domain_y / grid_dx)
const npml = 10

const rx_y = npml + 10
const rx_x_list = collect((npml+3):4:(nx-npml-3))

println("Domain: $(nx) × $(ny) cells, dx = $(grid_dx*1e3) mm")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Multi-source configurations
# ══════════════════════════════════════════════════════════════════════
src_x_list = [30, 65, 100, 135, 170]
nsrc = length(src_x_list)

configs = FDTDConfig[]
src_waveforms = Vector{Float64}[]

for (k, sx) in enumerate(src_x_list)
    cfg = create_config(
        nx=nx, ny=ny, dx=grid_dx, fc=fc_gpr, npml=npml,
        src_ix=sx, src_iy=rx_y,
        rx_iy=rx_y, rx_ix_list=rx_x_list,
    )
    push!(configs, cfg)
    push!(src_waveforms, create_source(cfg))
end

# ══════════════════════════════════════════════════════════════════════
# Build TRUE model (dispersive) — identical to run_fwi_large_domain.jl
# ══════════════════════════════════════════════════════════════════════
eps_inf_true = ones(nx, ny)
deps_true    = zeros(nx, ny)
tau_true     = zeros(nx, ny)
sigma_true   = zeros(nx, ny)

surface_j = npml + 15
layer1_top = surface_j
layer1_bot = surface_j + round(Int, 0.6 / grid_dx)
for j in layer1_top:min(layer1_bot, ny), i in 1:nx
    eps_inf_true[i, j] = 4.0
    deps_true[i, j]    = 4.0
    tau_true[i, j]     = 0.3e-9
    sigma_true[i, j]   = 0.005
end

layer2_top = layer1_bot + 1
for j in layer2_top:ny, i in 1:nx
    eps_inf_true[i, j] = 6.0
    deps_true[i, j]    = 12.0
    tau_true[i, j]     = 1.0e-9
    sigma_true[i, j]   = 0.02
end

pipe_cx = nx ÷ 2
pipe_cy = surface_j + round(Int, 0.4 / grid_dx)
pipe_r  = round(Int, 0.05 / grid_dx)

for j in 1:ny, i in 1:nx
    if (i - pipe_cx)^2 + (j - pipe_cy)^2 <= pipe_r^2
        eps_inf_true[i, j] = 15.0
        deps_true[i, j]    = 10.0
        tau_true[i, j]     = 0.5e-9
        sigma_true[i, j]   = 0.001
    end
end

# ══════════════════════════════════════════════════════════════════════
# Generate clean observed data from dispersive true model
# ══════════════════════════════════════════════════════════════════════
Random.seed!(42)

println("\nGenerating clean observed data...")
flush(stdout)
obs_datas_clean = Matrix{Float64}[]
for k in 1:nsrc
    od = run_forward!(configs[k], eps_inf_true, deps_true, tau_true, sigma_true, src_waveforms[k])
    push!(obs_datas_clean, od)
    @printf("  Source %d: max |Ez| = %.4e\n", k, maximum(abs.(od)))
    flush(stdout)
end

# ══════════════════════════════════════════════════════════════════════
# Initial model (dispersive, layered, no pipe) — same as dispersive FWI
# ══════════════════════════════════════════════════════════════════════
eps_inf_init = ones(nx, ny)
deps_init    = zeros(nx, ny)
tau_init     = zeros(nx, ny)
sigma_init   = zeros(nx, ny)

for j in layer1_top:min(layer1_bot, ny), i in 1:nx
    eps_inf_init[i, j] = 4.0
    deps_init[i, j]    = 4.0
    tau_init[i, j]     = 0.3e-9
    sigma_init[i, j]   = 0.005
end
for j in layer2_top:ny, i in 1:nx
    eps_inf_init[i, j] = 6.0
    deps_init[i, j]    = 12.0
    tau_init[i, j]     = 1.0e-9
    sigma_init[i, j]   = 0.02
end

# ══════════════════════════════════════════════════════════════════════
# Inversion region
# ══════════════════════════════════════════════════════════════════════
inv_x_lo = pipe_cx - 25
inv_x_hi = pipe_cx + 25
inv_y_lo = pipe_cy - 20
inv_y_hi = pipe_cy + 20

param_mask = falses(nx, ny)
for j in inv_y_lo:inv_y_hi, i in inv_x_lo:inv_x_hi
    param_mask[i, j] = true
end
n_params = count(param_mask)

# ══════════════════════════════════════════════════════════════════════
# Noise robustness loop: SNR = 40, 30, 20 dB
# ══════════════════════════════════════════════════════════════════════
snr_levels = [40, 30, 20]  # dB
summary_results = []

for snr_db in snr_levels
    println("\n" * "=" ^ 60)
    @printf("Noise robustness: SNR = %d dB\n", snr_db)
    println("=" ^ 60)
    flush(stdout)

    Random.seed!(42 + snr_db)  # reproducible noise per SNR level

    # Add noise to observed data
    obs_datas_noisy = Matrix{Float64}[]
    for k in 1:nsrc
        d_clean = obs_datas_clean[k]
        # SNR = 10*log10(||signal||² / ||noise||²)
        # ||noise||² = ||signal||² / 10^(SNR/10)
        signal_power = sum(d_clean .^ 2)
        noise_power = signal_power / (10.0^(snr_db / 10.0))
        noise_std = sqrt(noise_power / length(d_clean))
        noise = noise_std .* randn(size(d_clean))
        d_noisy = d_clean .+ noise
        push!(obs_datas_noisy, d_noisy)
        actual_snr = 10.0 * log10(signal_power / sum(noise .^ 2))
        @printf("  Source %d: noise_std=%.2e, actual_SNR=%.1f dB\n", k, noise_std, actual_snr)
        flush(stdout)
    end

    # Run FWI with noisy data
    t_fwi = time()
    result = run_fwi_multisource(configs, obs_datas_noisy, src_waveforms,
                                  eps_inf_init, deps_init, tau_init, sigma_init,
                                  param_mask;
                                  max_iter=20, param_type=:eps_inf,
                                  use_ad=true, verbose=true,
                                  lower_bound=1.0, upper_bound=25.0,
                                  lambda=1.0)
    t_fwi = time() - t_fwi
    @printf("\nSNR=%d dB FWI completed in %.1f s\n", snr_db, t_fwi)
    flush(stdout)

    # Save convergence
    conv_file = joinpath(datadir, "fwi_noisy_snr$(snr_db)db_convergence.csv")
    open(conv_file, "w") do io
        write(io, "# Noisy FWI convergence: SNR=$(snr_db)dB, $nsrc sources, lambda=1.0, seed=$(42+snr_db)\n")
        write(io, "iteration,loss,grad_norm\n")
        for k in 1:length(result.loss_history)
            @printf(io, "%d,%.12e,%.12e\n", k-1, result.loss_history[k],
                    result.grad_norm_history[k])
        end
    end
    println("  Saved: $conv_file")

    # Save 1D reconstruction
    recon_file = joinpath(datadir, "fwi_noisy_snr$(snr_db)db_reconstruction_1d.csv")
    open(recon_file, "w") do io
        write(io, "# 1D reconstruction (noisy FWI, SNR=$(snr_db)dB): pipe center slice\n")
        write(io, "depth_cm,eps_inf_true,eps_inf_initial,eps_inf_estimated\n")
        for j in inv_y_lo:inv_y_hi
            depth_cm = (j - surface_j) * grid_dx * 100.0
            @printf(io, "%.2f,%.6f,%.6f,%.6f\n",
                    depth_cm, eps_inf_true[pipe_cx, j], eps_inf_init[pipe_cx, j],
                    result.eps_inf_est[pipe_cx, j])
        end
    end
    println("  Saved: $recon_file")

    # Compute metrics
    eps_true_region = Float64[]
    eps_est_region = Float64[]
    for j in inv_y_lo:inv_y_hi, i in inv_x_lo:inv_x_hi
        push!(eps_true_region, eps_inf_true[i, j])
        push!(eps_est_region, result.eps_inf_est[i, j])
    end
    rmse = sqrt(mean((eps_true_region .- eps_est_region).^2))
    peak_est = maximum(result.eps_inf_est[inv_x_lo:inv_x_hi, inv_y_lo:inv_y_hi])
    peak_true = maximum(eps_inf_true[inv_x_lo:inv_x_hi, inv_y_lo:inv_y_hi])
    recovery = 100.0 * peak_est / peak_true
    reduction = 100.0 * (1.0 - result.loss_history[end] / result.loss_history[1])

    push!(summary_results, (snr_db, rmse, peak_est, recovery, reduction,
                             result.loss_history[1], result.loss_history[end]))

    @printf("  RMSE:          %.4f\n", rmse)
    @printf("  Peak ε∞:       %.2f (recovery %.1f%%)\n", peak_est, recovery)
    @printf("  Loss reduction: %.1f%%\n", reduction)
    flush(stdout)
end

# ══════════════════════════════════════════════════════════════════════
# Save summary table
# ══════════════════════════════════════════════════════════════════════
summary_file = joinpath(datadir, "fwi_noisy_summary.csv")
open(summary_file, "w") do io
    write(io, "# Noise robustness summary: dispersive FWI at 3 SNR levels, lambda=1.0\n")
    write(io, "snr_db,rmse,peak_eps_inf,peak_recovery_pct,loss_reduction_pct,loss_initial,loss_final\n")
    for (snr, rmse, peak, rec, red, l0, lf) in summary_results
        @printf(io, "%d,%.6f,%.4f,%.2f,%.2f,%.6e,%.6e\n", snr, rmse, peak, rec, red, l0, lf)
    end
end
println("\nSummary saved to: $summary_file")

println("\n=== Noise robustness study complete ===")
