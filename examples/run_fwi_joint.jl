# run_fwi_joint.jl — Joint ε∞ + σ multi-source FWI
#
# Demonstrates AD's extensibility: extending from single-parameter (ε∞)
# to joint two-parameter (ε∞ + σ) inversion required NO additional AD code —
# only the parameter packing and bounds were changed.
#
# Uses the same domain/setup as run_fwi_large_domain.jl (200×170) but inverts
# for both eps_inf and sigma simultaneously (param_type = :both).
#
# Outputs to paper/data/:
#   fwi_joint_convergence.csv
#   fwi_joint_reconstruction_2d_eps_true.csv
#   fwi_joint_reconstruction_2d_eps_initial.csv
#   fwi_joint_reconstruction_2d_eps_estimated.csv
#   fwi_joint_reconstruction_2d_sigma_true.csv
#   fwi_joint_reconstruction_2d_sigma_initial.csv
#   fwi_joint_reconstruction_2d_sigma_estimated.csv
#   fwi_joint_reconstruction_1d.csv

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
# Domain setup — same as run_fwi_large_domain.jl
# ══════════════════════════════════════════════════════════════════════
const domain_x = 1.0    # [m]
const domain_y = 0.85   # [m]
const grid_dx  = 0.005  # [m] (5 mm)
const fc_gpr   = 500e6  # [Hz] center frequency

const nx = round(Int, domain_x / grid_dx)   # 200
const ny = round(Int, domain_y / grid_dx)    # 170
const npml = 10

# Receiver line at surface
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
# Build material model (true model with pipe)
# ══════════════════════════════════════════════════════════════════════
eps_inf_true = ones(nx, ny)
deps_true    = zeros(nx, ny)
tau_true     = zeros(nx, ny)
sigma_true   = zeros(nx, ny)

surface_j = npml + 15

# Layer 1: sandy soil
layer1_top = surface_j
layer1_bot = surface_j + round(Int, 0.6 / grid_dx)
for j in layer1_top:min(layer1_bot, ny), i in 1:nx
    eps_inf_true[i, j] = 4.0
    deps_true[i, j]    = 4.0
    tau_true[i, j]     = 0.3e-9
    sigma_true[i, j]   = 0.005
end

# Layer 2: clay
layer2_top = layer1_bot + 1
for j in layer2_top:ny, i in 1:nx
    eps_inf_true[i, j] = 6.0
    deps_true[i, j]    = 12.0
    tau_true[i, j]     = 1.0e-9
    sigma_true[i, j]   = 0.02
end

# Buried pipe
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
println("  True sigma: background=0.005, pipe=0.001, clay=0.02")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Initial model: layered soil WITHOUT pipe (known geology, unknown target)
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
# Generate observed data
# ══════════════════════════════════════════════════════════════════════
println("\nGenerating observed data for $nsrc sources...")
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
# Inversion region (same as single-parameter)
# ══════════════════════════════════════════════════════════════════════
inv_x_lo = pipe_cx - 25
inv_x_hi = pipe_cx + 25
inv_y_lo = pipe_cy - 20
inv_y_hi = pipe_cy + 20

param_mask = falses(nx, ny)
for j in inv_y_lo:inv_y_hi, i in inv_x_lo:inv_x_hi
    param_mask[i, j] = true
end
n_cells = count(param_mask)
n_params = 2 * n_cells  # two parameters per cell
println("\nInversion region: x=$inv_x_lo:$inv_x_hi, y=$inv_y_lo:$inv_y_hi")
println("  Cells: $n_cells, Parameters: $n_params (joint ε∞ + σ)")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Run joint multi-source FWI
# ══════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("Running JOINT ε∞ + σ multi-source FWI ($nsrc sources, $n_params params)")
println("=" ^ 60)
flush(stdout)

# Regularization weights:
#   lambda = 1.0 for ε∞ (same as single-parameter case)
#   lambda_sigma = 1e4 for σ (scaled by ~(ε∞_range/σ_range)² to balance penalties)
t_fwi = time()
result = run_fwi_multisource(configs, obs_datas, src_waveforms,
                              eps_inf_init, deps_init, tau_init, sigma_init,
                              param_mask;
                              max_iter=50, param_type=:both,
                              use_ad=true, verbose=true,
                              lower_bound=1.0, upper_bound=25.0,
                              lower_bound_sigma=0.0, upper_bound_sigma=0.1,
                              lambda=1.0, lambda_sigma=1e4)
t_fwi = time() - t_fwi
@printf("\nJoint FWI completed in %.1f s (%d iterations)\n", t_fwi, result.n_iter)
@printf("  Initial loss: %.6e\n", result.loss_history[1])
@printf("  Final loss:   %.6e\n", result.loss_history[end])
@printf("  Reduction:    %.4f%%\n",
        100.0 * (1.0 - result.loss_history[end] / result.loss_history[1]))
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════

# 1. Convergence history
conv_file = joinpath(datadir, "fwi_joint_convergence.csv")
open(conv_file, "w") do io
    write(io, "# Joint FWI convergence: $nsrc sources, $n_params params (eps_inf+sigma), AD, seed=42\n")
    write(io, "iteration,loss,grad_norm\n")
    for k in 1:length(result.loss_history)
        @printf(io, "%d,%.12e,%.12e\n",
                k-1,
                result.loss_history[k],
                result.grad_norm_history[k])
    end
end
println("Convergence saved to: $conv_file")

# 2-4. 2D eps_inf maps (inversion region)
for (tag, emap) in [("true", eps_inf_true), ("initial", eps_inf_init), ("estimated", result.eps_inf_est)]
    fname = joinpath(datadir, "fwi_joint_reconstruction_2d_eps_$(tag).csv")
    open(fname, "w") do io
        write(io, "# 2D eps_inf map ($tag): joint inversion region, seed=42\n")
        write(io, "x_cm,depth_cm,eps_inf\n")
        for j in inv_y_lo:inv_y_hi
            depth_cm = (j - surface_j) * grid_dx * 100.0
            for i in inv_x_lo:inv_x_hi
                x_cm = i * grid_dx * 100.0
                @printf(io, "%.2f,%.2f,%.6f\n", x_cm, depth_cm, emap[i, j])
            end
        end
    end
    println("Saved: $fname")
end

# 5-7. 2D sigma maps (inversion region)
for (tag, smap) in [("true", sigma_true), ("initial", sigma_init), ("estimated", result.sigma_est)]
    fname = joinpath(datadir, "fwi_joint_reconstruction_2d_sigma_$(tag).csv")
    open(fname, "w") do io
        write(io, "# 2D sigma map ($tag): joint inversion region, seed=42\n")
        write(io, "x_cm,depth_cm,sigma\n")
        for j in inv_y_lo:inv_y_hi
            depth_cm = (j - surface_j) * grid_dx * 100.0
            for i in inv_x_lo:inv_x_hi
                x_cm = i * grid_dx * 100.0
                @printf(io, "%.2f,%.2f,%.6f\n", x_cm, depth_cm, smap[i, j])
            end
        end
    end
    println("Saved: $fname")
end

# 8. 1D reconstruction slice (vertical through pipe center)
recon1d_file = joinpath(datadir, "fwi_joint_reconstruction_1d.csv")
open(recon1d_file, "w") do io
    write(io, "# 1D joint reconstruction: vertical slice at x=$(pipe_cx*grid_dx) m through pipe center\n")
    write(io, "depth_cm,eps_inf_true,eps_inf_initial,eps_inf_estimated,sigma_true,sigma_initial,sigma_estimated\n")
    for j in inv_y_lo:inv_y_hi
        depth_cm = (j - surface_j) * grid_dx * 100.0
        @printf(io, "%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                depth_cm,
                eps_inf_true[pipe_cx, j], eps_inf_init[pipe_cx, j], result.eps_inf_est[pipe_cx, j],
                sigma_true[pipe_cx, j], sigma_init[pipe_cx, j], result.sigma_est[pipe_cx, j])
    end
end
println("Saved: $recon1d_file")

# Summary statistics — ε∞
eps_true_region = Float64[]
eps_est_region = Float64[]
for j in inv_y_lo:inv_y_hi, i in inv_x_lo:inv_x_hi
    push!(eps_true_region, eps_inf_true[i, j])
    push!(eps_est_region, result.eps_inf_est[i, j])
end
rmse_eps = sqrt(mean((eps_true_region .- eps_est_region).^2))
peak_eps_est = maximum(result.eps_inf_est[inv_x_lo:inv_x_hi, inv_y_lo:inv_y_hi])
peak_eps_true = maximum(eps_inf_true[inv_x_lo:inv_x_hi, inv_y_lo:inv_y_hi])

# Summary statistics — σ
sig_true_region = Float64[]
sig_est_region = Float64[]
for j in inv_y_lo:inv_y_hi, i in inv_x_lo:inv_x_hi
    push!(sig_true_region, sigma_true[i, j])
    push!(sig_est_region, result.sigma_est[i, j])
end
rmse_sigma = sqrt(mean((sig_true_region .- sig_est_region).^2))

@printf("\nReconstruction quality (ε∞):\n")
@printf("  Peak ε∞ true:      %.2f\n", peak_eps_true)
@printf("  Peak ε∞ estimated: %.2f\n", peak_eps_est)
@printf("  Peak recovery:     %.1f%%\n", 100.0 * peak_eps_est / peak_eps_true)
@printf("  RMSE(ε∞):          %.4f\n", rmse_eps)

@printf("\nReconstruction quality (σ):\n")
@printf("  True σ (pipe):     %.4f S/m\n", sigma_true[pipe_cx, pipe_cy])
@printf("  Est σ (pipe):      %.4f S/m\n", result.sigma_est[pipe_cx, pipe_cy])
@printf("  True σ (bg):       %.4f S/m\n", sigma_true[inv_x_lo, inv_y_lo])
@printf("  RMSE(σ):           %.6f S/m\n", rmse_sigma)
flush(stdout)

println("\n=== Joint ε∞ + σ FWI complete ===")
