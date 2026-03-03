# validate_gradients.jl — AD gradient verification and simple FWI inversion
#
# 1. Verifies AD gradients against finite-difference gradients on a small problem
# 2. Runs a simple FWI inversion on a 1D layered model
# 3. Saves CSV data and generates plots

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using GPRADFWI
using DelimitedFiles
using Printf
using Random
using LinearAlgebra
using Statistics

Random.seed!(42)

datadir = joinpath(@__DIR__, "..", "..", "paper", "data")
mkpath(datadir)

# ══════════════════════════════════════════════════════════════════════
# Part 1: Gradient Verification on Small Problem
# ══════════════════════════════════════════════════════════════════════
println("=" ^ 60)
println("Part 1: Gradient Verification (AD vs Finite Differences)")
println("=" ^ 60)

# Small domain for gradient testing
const nx_small = 60
const ny_small = 50
const dx_small = 0.01  # 1 cm
const fc_small = 300e6  # 300 MHz (lower freq → fewer time steps needed)
const npml_small = 8

# Receiver line
rx_list_small = collect((npml_small+2):3:(nx_small-npml_small-2))
rx_y_small = npml_small + 5
src_x_small = nx_small ÷ 2
src_y_small = rx_y_small

config_small = create_config(
    nx=nx_small, ny=ny_small, dx=dx_small, fc=fc_small, npml=npml_small,
    src_ix=src_x_small, src_iy=src_y_small,
    rx_iy=rx_y_small, rx_ix_list=rx_list_small,
    nt=300,  # short simulation for gradient testing
)

# True model: uniform background with a small anomaly
eps_inf_true = 3.0 * ones(nx_small, ny_small)
deps_true    = zeros(nx_small, ny_small)
tau_true     = zeros(nx_small, ny_small)
sigma_true   = 0.001 * ones(nx_small, ny_small)

# Add anomaly
anomaly_i = nx_small ÷ 2
anomaly_j = ny_small ÷ 2
anomaly_r = 3
for j in 1:ny_small, i in 1:nx_small
    if (i - anomaly_i)^2 + (j - anomaly_j)^2 <= anomaly_r^2
        eps_inf_true[i, j] = 6.0
    end
end

# Generate observed data
src_wf_small = create_source(config_small)
println("Generating observed data...")
obs_data = run_forward!(config_small, eps_inf_true, deps_true, tau_true, sigma_true, src_wf_small)
println("  Max |Ez| observed: $(@sprintf("%.6e", maximum(abs.(obs_data))))")

# Define inversion mask: only invert a small region around anomaly
param_mask = falses(nx_small, ny_small)
inv_region = 6  # cells around anomaly
for j in (anomaly_j-inv_region):(anomaly_j+inv_region)
    for i in (anomaly_i-inv_region):(anomaly_i+inv_region)
        if 1 <= i <= nx_small && 1 <= j <= ny_small
            param_mask[i, j] = true
        end
    end
end
n_params = count(param_mask)
println("Inversion parameters: $n_params cells")

# Background (initial) model
eps_inf_bg = 3.0 * ones(nx_small, ny_small)
sigma_bg = 0.001 * ones(nx_small, ny_small)

# Objective function wrapper
function obj_eps(x_flat)
    return forward_misfit(x_flat, config_small, obs_data, src_wf_small,
                          eps_inf_bg, deps_true, tau_true, sigma_bg,
                          param_mask, :eps_inf)
end

# Pack initial parameters
x0 = Float64[]
for j in 1:ny_small, i in 1:nx_small
    if param_mask[i, j]
        push!(x0, eps_inf_bg[i, j])
    end
end

println("\nComputing FD gradient (h=1e-5)...")
t_fd = time()
grad_fd = fd_gradient(obj_eps, x0; h=1e-5)
t_fd = time() - t_fd
@printf("FD gradient computed in %.2f s\n", t_fd)

println("Computing AD gradient...")
t_ad = time()
grad_ad = ad_gradient(obj_eps, x0)
t_ad = time() - t_ad
@printf("AD gradient computed in %.2f s\n", t_ad)

# Compare gradients
rel_err = norm(grad_ad - grad_fd) / max(norm(grad_fd), 1e-30)
max_abs_err = maximum(abs.(grad_ad - grad_fd))
@printf("\nGradient comparison:\n")
@printf("  ||g_AD - g_FD|| / ||g_FD|| = %.6e\n", rel_err)
@printf("  max |g_AD - g_FD|           = %.6e\n", max_abs_err)
@printf("  ||g_FD||                    = %.6e\n", norm(grad_fd))
@printf("  ||g_AD||                    = %.6e\n", norm(grad_ad))

# Per-parameter comparison
per_param_err = abs.(grad_ad .- grad_fd) ./ max.(abs.(grad_fd), 1e-30)
@printf("  Median per-param rel error  = %.6e\n", median(per_param_err))
@printf("  Max per-param rel error     = %.6e\n", maximum(per_param_err))

# Save gradient comparison CSV
grad_file = joinpath(datadir, "gradient_comparison.csv")
open(grad_file, "w") do io
    write(io, "# Gradient verification: AD vs FD, n_params=$n_params, seed=42\n")
    write(io, "param_index,grad_fd,grad_ad,abs_error,rel_error\n")
    for k in 1:length(x0)
        ae = abs(grad_ad[k] - grad_fd[k])
        re = ae / max(abs(grad_fd[k]), 1e-30)
        @printf(io, "%d,%.12e,%.12e,%.6e,%.6e\n", k, grad_fd[k], grad_ad[k], ae, re)
    end
end
println("Gradient comparison saved to: $grad_file")

# ══════════════════════════════════════════════════════════════════════
# Part 2: Simple FWI Inversion (eps_inf only)
# ══════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("Part 2: FWI Inversion (permittivity reconstruction)")
println("=" ^ 60)

result = run_fwi(config_small, obs_data, src_wf_small,
                 eps_inf_bg, deps_true, tau_true, sigma_bg, param_mask;
                 max_iter=25, param_type=:eps_inf, use_ad=false, verbose=true)

println("\nFWI completed: $(result.n_iter) iterations")
@printf("  Initial loss: %.6e\n", result.loss_history[1])
@printf("  Final loss:   %.6e\n", result.loss_history[end])
@printf("  Reduction:    %.2f%%\n",
        100.0 * (1.0 - result.loss_history[end] / result.loss_history[1]))

# Compare reconstruction
eps_true_region = Float64[]
eps_est_region = Float64[]
for j in 1:ny_small, i in 1:nx_small
    if param_mask[i, j]
        push!(eps_true_region, eps_inf_true[i, j])
        push!(eps_est_region, result.eps_inf_est[i, j])
    end
end

rmse = sqrt(mean((eps_true_region .- eps_est_region).^2))
@printf("  RMSE(ε∞): %.4f\n", rmse)

# Save inversion convergence CSV
conv_file = joinpath(datadir, "inversion_convergence.csv")
open(conv_file, "w") do io
    write(io, "# FWI convergence: eps_inf inversion, $(result.n_iter) iters, seed=42\n")
    write(io, "iteration,loss,grad_norm\n")
    for k in 1:length(result.loss_history)
        @printf(io, "%d,%.12e,%.12e\n", k-1, result.loss_history[k],
                result.grad_norm_history[k])
    end
end
println("Convergence saved to: $conv_file")

# Save reconstructed model profile (vertical slice)
recon_file = joinpath(datadir, "inversion_reconstruction.csv")
open(recon_file, "w") do io
    write(io, "# Reconstructed permittivity: vertical slice at x=$(anomaly_i)\n")
    write(io, "y_index,depth_cm,eps_inf_true,eps_inf_estimated,eps_inf_initial\n")
    for j in max(1, anomaly_j-inv_region):min(ny_small, anomaly_j+inv_region)
        depth_cm = (j - rx_y_small) * dx_small * 100.0
        @printf(io, "%d,%.2f,%.6f,%.6f,%.6f\n",
                j, depth_cm,
                eps_inf_true[anomaly_i, j],
                result.eps_inf_est[anomaly_i, j],
                eps_inf_bg[anomaly_i, j])
    end
end
println("Reconstruction saved to: $recon_file")

println("\n=== Gradient validation & inversion complete ===")
