# run_fwi_nondispersive.jl — Non-dispersive FWI comparison (C1)
#
# Runs the same multi-source FWI as run_fwi_large_domain.jl but with
# deps=0, tau=0 in the forward model. The observed data remains from the
# dispersive true model. This simulates the realistic scenario where the
# subsurface IS dispersive but the forward model ignores dispersion.
#
# Outputs to paper/data/:
#   fwi_nondispersive_convergence.csv
#   fwi_nondispersive_reconstruction_2d_estimated.csv
#   fwi_nondispersive_reconstruction_1d.csv

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
# Domain setup — identical to run_fwi_large_domain.jl
# ══════════════════════════════════════════════════════════════════════
const domain_x = 1.0
const domain_y = 0.85
const grid_dx  = 0.005
const fc_gpr   = 500e6

const nx = round(Int, domain_x / grid_dx)   # 200
const ny = round(Int, domain_y / grid_dx)    # 170
const npml = 10

const rx_y = npml + 10
const rx_x_list = collect((npml+3):4:(nx-npml-3))

println("Domain: $(nx) × $(ny) cells, dx = $(grid_dx*1e3) mm")
println("Receivers: $(length(rx_x_list)) at y-index $rx_y")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Multi-source configurations: 5 transmitters
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
    @printf("  Source %d: x-index=%d (%.2f m), nt=%d\n", k, sx, sx * grid_dx, cfg.nt)
    flush(stdout)
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
    r2 = (i - pipe_cx)^2 + (j - pipe_cy)^2
    if r2 <= pipe_r^2
        eps_inf_true[i, j] = 15.0
        deps_true[i, j]    = 10.0
        tau_true[i, j]     = 0.5e-9
        sigma_true[i, j]   = 0.001
    end
end

println("\nPipe: center=($pipe_cx, $pipe_cy), r=$pipe_r cells")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Generate observed data from DISPERSIVE true model
# (Same data as the dispersive FWI — the ground truth is dispersive)
# ══════════════════════════════════════════════════════════════════════
println("\nGenerating observed data from dispersive true model...")
flush(stdout)
obs_datas = Matrix{Float64}[]
for k in 1:nsrc
    t0 = time()
    od = run_forward!(configs[k], eps_inf_true, deps_true, tau_true, sigma_true, src_waveforms[k])
    dt_sec = time() - t0
    push!(obs_datas, od)
    @printf("  Source %d: max |Ez| = %.4e, time = %.1f s\n", k, maximum(abs.(od)), dt_sec)
    flush(stdout)
end

# ══════════════════════════════════════════════════════════════════════
# Initial model: layered soil WITHOUT pipe, NON-DISPERSIVE (deps=0, tau=0)
# This is the key difference: the forward model ignores dispersion
# ══════════════════════════════════════════════════════════════════════
eps_inf_init = ones(nx, ny)
deps_init    = zeros(nx, ny)   # NON-DISPERSIVE: deps=0
tau_init     = zeros(nx, ny)   # NON-DISPERSIVE: tau=0
sigma_init   = zeros(nx, ny)

for j in layer1_top:min(layer1_bot, ny), i in 1:nx
    eps_inf_init[i, j] = 4.0
    # deps_init stays 0 — non-dispersive
    # tau_init stays 0  — non-dispersive
    sigma_init[i, j]   = 0.005
end
for j in layer2_top:ny, i in 1:nx
    eps_inf_init[i, j] = 6.0
    # deps_init stays 0 — non-dispersive
    # tau_init stays 0  — non-dispersive
    sigma_init[i, j]   = 0.02
end

println("\nNON-DISPERSIVE forward model: deps=0, tau=0 everywhere")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Define inversion region (same as dispersive FWI)
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
println("Inversion region: $n_params parameters")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Run multi-source FWI with NON-DISPERSIVE forward model
# ══════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("Running NON-DISPERSIVE multi-source FWI ($nsrc sources, $n_params params)")
println("=" ^ 60)
flush(stdout)

t_fwi = time()
result = run_fwi_multisource(configs, obs_datas, src_waveforms,
                              eps_inf_init, deps_init, tau_init, sigma_init,
                              param_mask;
                              max_iter=50, param_type=:eps_inf,
                              use_ad=true, verbose=true,
                              lower_bound=1.0, upper_bound=25.0,
                              lambda=1.0)
t_fwi = time() - t_fwi
@printf("\nNon-dispersive FWI completed in %.1f s (%d iterations)\n", t_fwi, result.n_iter)
@printf("  Initial loss: %.6e\n", result.loss_history[1])
@printf("  Final loss:   %.6e\n", result.loss_history[end])
@printf("  Reduction:    %.4f%%\n",
        100.0 * (1.0 - result.loss_history[end] / result.loss_history[1]))
flush(stdout)

# Report regularization breakdown
idx_map = GPRADFWI._build_idx_map(param_mask)
x_final = Float64[]
for j in 1:ny, i in 1:nx
    if param_mask[i, j]
        push!(x_final, result.eps_inf_est[i, j])
    end
end
reg_penalty = GPRADFWI.tikhonov_penalty(x_final, idx_map, param_mask)
reg_term = 1.0 * reg_penalty
data_misfit = result.loss_history[end] - reg_term
@printf("  Data misfit:  %.6e\n", data_misfit)
@printf("  Reg term:     %.6e\n", reg_term)
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════

# 1. Convergence history
conv_file = joinpath(datadir, "fwi_nondispersive_convergence.csv")
open(conv_file, "w") do io
    write(io, "# Non-dispersive FWI convergence: $nsrc sources, $n_params params, AD, deps=0, tau=0, seed=42\n")
    write(io, "iteration,loss,grad_norm\n")
    for k in 1:length(result.loss_history)
        @printf(io, "%d,%.12e,%.12e\n", k-1, result.loss_history[k],
                result.grad_norm_history[k])
    end
end
println("Convergence saved to: $conv_file")

# 2. 2D reconstruction map (estimated only — true and initial are same as dispersive)
fname = joinpath(datadir, "fwi_nondispersive_reconstruction_2d_estimated.csv")
open(fname, "w") do io
    write(io, "# 2D eps_inf map (estimated, non-dispersive FWI): inversion region, seed=42\n")
    write(io, "x_cm,depth_cm,eps_inf\n")
    for j in inv_y_lo:inv_y_hi
        depth_cm = (j - surface_j) * grid_dx * 100.0
        for i in inv_x_lo:inv_x_hi
            x_cm = i * grid_dx * 100.0
            @printf(io, "%.2f,%.2f,%.6f\n", x_cm, depth_cm, result.eps_inf_est[i, j])
        end
    end
end
println("Saved: $fname")

# 3. 1D reconstruction slice
recon1d_file = joinpath(datadir, "fwi_nondispersive_reconstruction_1d.csv")
open(recon1d_file, "w") do io
    write(io, "# 1D reconstruction (non-dispersive FWI): vertical slice at pipe center\n")
    write(io, "depth_cm,eps_inf_true,eps_inf_initial,eps_inf_estimated\n")
    for j in inv_y_lo:inv_y_hi
        depth_cm = (j - surface_j) * grid_dx * 100.0
        @printf(io, "%.2f,%.6f,%.6f,%.6f\n",
                depth_cm, eps_inf_true[pipe_cx, j], eps_inf_init[pipe_cx, j],
                result.eps_inf_est[pipe_cx, j])
    end
end
println("Saved: $recon1d_file")

# Summary statistics
eps_true_region = Float64[]
eps_est_region = Float64[]
for j in inv_y_lo:inv_y_hi, i in inv_x_lo:inv_x_hi
    push!(eps_true_region, eps_inf_true[i, j])
    push!(eps_est_region, result.eps_inf_est[i, j])
end
rmse = sqrt(mean((eps_true_region .- eps_est_region).^2))
peak_est = maximum(result.eps_inf_est[inv_x_lo:inv_x_hi, inv_y_lo:inv_y_hi])
peak_true = maximum(eps_inf_true[inv_x_lo:inv_x_hi, inv_y_lo:inv_y_hi])

@printf("\nNon-dispersive reconstruction quality:\n")
@printf("  Peak ε∞ true:      %.2f\n", peak_true)
@printf("  Peak ε∞ estimated: %.2f\n", peak_est)
@printf("  Peak recovery:     %.1f%%\n", 100.0 * peak_est / peak_true)
@printf("  RMSE:              %.4f\n", rmse)
flush(stdout)

println("\n=== Non-dispersive FWI comparison complete ===")
