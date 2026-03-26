# run_fwi_two_anomaly.jl — Second subsurface model: two buried anomalies
#
# Demonstrates the AD-based FWI framework on a more complex scenario than the
# single-pipe model. Two anomalies at different depths and horizontal positions
# in the same two-layer dispersive soil:
#   1. High-permittivity pipe (ε∞=15, Δε=10) at shallow depth
#   2. Low-permittivity void (ε∞=2, Δε=0.5) at deeper location
#
# Uses the reduced domain (120×100, dx=5mm) with 5 sources, 40 iterations.
#
# Outputs to paper/data/:
#   fwi_two_anomaly_convergence.csv
#   fwi_two_anomaly_reconstruction_2d_true.csv
#   fwi_two_anomaly_reconstruction_2d_initial.csv
#   fwi_two_anomaly_reconstruction_2d_estimated.csv
#   fwi_two_anomaly_reconstruction_1d_pipe.csv
#   fwi_two_anomaly_reconstruction_1d_void.csv

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
# Domain setup — reduced domain for tractable computation
# ══════════════════════════════════════════════════════════════════════
const domain_x = 0.60
const domain_y = 0.50
const grid_dx  = 0.005
const fc_gpr   = 500e6

const nx = round(Int, domain_x / grid_dx)   # 120
const ny = round(Int, domain_y / grid_dx)    # 100
const npml = 10

const rx_y = npml + 8
const rx_x_list = collect((npml+3):3:(nx-npml-3))

println("Domain: $(nx) × $(ny) cells, dx = $(grid_dx*1e3) mm")
println("Receivers: $(length(rx_x_list)) at y-index $rx_y")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Multi-source configurations: 5 transmitters
# ══════════════════════════════════════════════════════════════════════
src_x_list = [20, 40, 60, 80, 100]
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
# Build TRUE model: two anomalies in dispersive two-layer soil
# ══════════════════════════════════════════════════════════════════════
eps_inf_true = ones(nx, ny)
deps_true    = zeros(nx, ny)
tau_true     = zeros(nx, ny)
sigma_true   = zeros(nx, ny)

surface_j = npml + 10

# Layer 1: sandy soil (0 to 0.25m depth)
layer1_top = surface_j
layer1_bot = surface_j + round(Int, 0.25 / grid_dx)
for j in layer1_top:min(layer1_bot, ny), i in 1:nx
    eps_inf_true[i, j] = 4.0
    deps_true[i, j]    = 4.0
    tau_true[i, j]     = 0.3e-9
    sigma_true[i, j]   = 0.005
end

# Layer 2: clay (0.25m to bottom)
layer2_top = layer1_bot + 1
for j in layer2_top:ny, i in 1:nx
    eps_inf_true[i, j] = 6.0
    deps_true[i, j]    = 12.0
    tau_true[i, j]     = 1.0e-9
    sigma_true[i, j]   = 0.02
end

# Anomaly 1: water-filled pipe at shallow depth in layer 1
pipe_cx = round(Int, 0.20 / grid_dx)    # x = 0.20m (left side)
pipe_cy = surface_j + round(Int, 0.15 / grid_dx)   # 0.15m depth
pipe_r  = round(Int, 0.025 / grid_dx)   # 2.5cm radius (5 cells)

for j in 1:ny, i in 1:nx
    r2 = (i - pipe_cx)^2 + (j - pipe_cy)^2
    if r2 <= pipe_r^2
        eps_inf_true[i, j] = 15.0
        deps_true[i, j]    = 10.0
        tau_true[i, j]     = 0.5e-9
        sigma_true[i, j]   = 0.001
    end
end

# Anomaly 2: air void at deeper location in layer 2
void_cx = round(Int, 0.40 / grid_dx)    # x = 0.40m (right side)
void_cy = surface_j + round(Int, 0.30 / grid_dx)   # 0.30m depth (in layer 2)
void_r  = round(Int, 0.02 / grid_dx)    # 2cm radius (4 cells)

for j in 1:ny, i in 1:nx
    r2 = (i - void_cx)^2 + (j - void_cy)^2
    if r2 <= void_r^2
        eps_inf_true[i, j] = 2.0
        deps_true[i, j]    = 0.5
        tau_true[i, j]     = 0.1e-9
        sigma_true[i, j]   = 0.0005
    end
end

@printf("\nPipe: center=(%d, %d) = (%.2fm, %.3fm depth), r=%d cells\n",
        pipe_cx, pipe_cy, pipe_cx*grid_dx, (pipe_cy-surface_j)*grid_dx, pipe_r)
@printf("Void: center=(%d, %d) = (%.2fm, %.3fm depth), r=%d cells\n",
        void_cx, void_cy, void_cx*grid_dx, (void_cy-surface_j)*grid_dx, void_r)
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Initial model: layered soil WITHOUT anomalies (known geology, unknown targets)
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
# Generate observed data for each source
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
# Define inversion region (rectangle covering both anomalies)
# ══════════════════════════════════════════════════════════════════════
inv_x_lo = min(pipe_cx, void_cx) - 15
inv_x_hi = max(pipe_cx, void_cx) + 15
inv_y_lo = min(pipe_cy, void_cy) - 15
inv_y_hi = max(pipe_cy, void_cy) + 15

# Clamp to domain
inv_x_lo = max(inv_x_lo, npml + 2)
inv_x_hi = min(inv_x_hi, nx - npml - 1)
inv_y_lo = max(inv_y_lo, npml + 2)
inv_y_hi = min(inv_y_hi, ny - npml - 1)

param_mask = falses(nx, ny)
for j in inv_y_lo:inv_y_hi, i in inv_x_lo:inv_x_hi
    param_mask[i, j] = true
end
n_params = count(param_mask)
println("\nInversion region: x=$inv_x_lo:$inv_x_hi, y=$inv_y_lo:$inv_y_hi")
println("  Size: $(inv_x_hi-inv_x_lo+1) × $(inv_y_hi-inv_y_lo+1) = $n_params parameters")
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Run multi-source FWI
# ══════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("Running TWO-ANOMALY FWI ($nsrc sources, $n_params params, AD)")
println("=" ^ 60)
flush(stdout)

t_fwi = time()
result = run_fwi_multisource(configs, obs_datas, src_waveforms,
                              eps_inf_init, deps_init, tau_init, sigma_init,
                              param_mask;
                              max_iter=40, param_type=:eps_inf,
                              use_ad=true, verbose=true,
                              lower_bound=1.0, upper_bound=25.0,
                              lambda=1.0)
t_fwi = time() - t_fwi
@printf("\nTwo-anomaly FWI completed in %.1f s (%d iterations)\n", t_fwi, result.n_iter)
@printf("  Initial loss: %.6e\n", result.loss_history[1])
@printf("  Final loss:   %.6e\n", result.loss_history[end])
@printf("  Reduction:    %.4f%%\n",
        100.0 * (1.0 - result.loss_history[end] / result.loss_history[1]))
flush(stdout)

# ══════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════

# 1. Convergence history
conv_file = joinpath(datadir, "fwi_two_anomaly_convergence.csv")
open(conv_file, "w") do io
    write(io, "# Two-anomaly FWI convergence: $nsrc sources, $n_params params, AD, seed=42\n")
    write(io, "iteration,loss,grad_norm\n")
    for k in 1:length(result.loss_history)
        @printf(io, "%d,%.12e,%.12e\n",
                k-1,
                result.loss_history[k],
                result.grad_norm_history[k])
    end
end
println("Convergence saved to: $conv_file")

# 2-4. 2D reconstruction maps (inversion region only)
for (tag, emap) in [("true", eps_inf_true), ("initial", eps_inf_init), ("estimated", result.eps_inf_est)]
    fname = joinpath(datadir, "fwi_two_anomaly_reconstruction_2d_$(tag).csv")
    open(fname, "w") do io
        write(io, "# 2D eps_inf map ($tag): inversion region, two-anomaly model, seed=42\n")
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

# 5. 1D reconstruction slice through pipe center
recon_pipe_file = joinpath(datadir, "fwi_two_anomaly_reconstruction_1d_pipe.csv")
open(recon_pipe_file, "w") do io
    write(io, "# 1D reconstruction: vertical slice at pipe center x=$(pipe_cx*grid_dx) m\n")
    write(io, "depth_cm,eps_inf_true,eps_inf_initial,eps_inf_estimated\n")
    for j in inv_y_lo:inv_y_hi
        depth_cm = (j - surface_j) * grid_dx * 100.0
        @printf(io, "%.2f,%.6f,%.6f,%.6f\n",
                depth_cm, eps_inf_true[pipe_cx, j], eps_inf_init[pipe_cx, j],
                result.eps_inf_est[pipe_cx, j])
    end
end
println("Saved: $recon_pipe_file")

# 6. 1D reconstruction slice through void center
recon_void_file = joinpath(datadir, "fwi_two_anomaly_reconstruction_1d_void.csv")
open(recon_void_file, "w") do io
    write(io, "# 1D reconstruction: vertical slice at void center x=$(void_cx*grid_dx) m\n")
    write(io, "depth_cm,eps_inf_true,eps_inf_initial,eps_inf_estimated\n")
    for j in inv_y_lo:inv_y_hi
        depth_cm = (j - surface_j) * grid_dx * 100.0
        @printf(io, "%.2f,%.6f,%.6f,%.6f\n",
                depth_cm, eps_inf_true[void_cx, j], eps_inf_init[void_cx, j],
                result.eps_inf_est[void_cx, j])
    end
end
println("Saved: $recon_void_file")

# Summary statistics
eps_true_region = Float64[]
eps_est_region = Float64[]
for j in inv_y_lo:inv_y_hi, i in inv_x_lo:inv_x_hi
    push!(eps_true_region, eps_inf_true[i, j])
    push!(eps_est_region, result.eps_inf_est[i, j])
end
rmse = sqrt(mean((eps_true_region .- eps_est_region).^2))

# Per-anomaly metrics
eps_pipe_true = eps_inf_true[pipe_cx, pipe_cy]
eps_pipe_est  = result.eps_inf_est[pipe_cx, pipe_cy]
eps_void_true = eps_inf_true[void_cx, void_cy]
eps_void_est  = result.eps_inf_est[void_cx, void_cy]

# Peak permittivity in pipe region
pipe_peak = maximum(result.eps_inf_est[max(pipe_cx-pipe_r,inv_x_lo):min(pipe_cx+pipe_r,inv_x_hi),
                                        max(pipe_cy-pipe_r,inv_y_lo):min(pipe_cy+pipe_r,inv_y_hi)])
# Minimum permittivity in void region
void_min = minimum(result.eps_inf_est[max(void_cx-void_r,inv_x_lo):min(void_cx+void_r,inv_x_hi),
                                       max(void_cy-void_r,inv_y_lo):min(void_cy+void_r,inv_y_hi)])

@printf("\nTwo-anomaly reconstruction quality:\n")
@printf("  Overall RMSE: %.4f\n", rmse)
@printf("  Pipe: true=%.1f, center_est=%.2f, peak_est=%.2f, recovery=%.1f%%\n",
        eps_pipe_true, eps_pipe_est, pipe_peak, 100.0 * pipe_peak / eps_pipe_true)
@printf("  Void: true=%.1f, center_est=%.2f, min_est=%.2f\n",
        eps_void_true, eps_void_est, void_min)
@printf("  Objective reduction: %.2f%%\n",
        100.0 * (1.0 - result.loss_history[end] / result.loss_history[1]))
flush(stdout)

println("\n=== Two-anomaly FWI complete ===")
