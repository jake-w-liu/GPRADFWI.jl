# lambda_sweep.jl — Quick sweep to find optimal Tikhonov regularization lambda
#
# Runs 5-iteration FWI for each lambda in {0.1, 1.0, 10.0, 100.0} and reports
# the data misfit vs regularization breakdown.
# Uses identical setup to run_fwi_large_domain.jl.

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

# ══════════════════════════════════════════════════════════════════════
# Domain setup (identical to run_fwi_large_domain.jl)
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

println("Lambda sweep — Domain: $(nx) × $(ny), dx = $(grid_dx*1e3) mm")
flush(stdout)

# Multi-source configs
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
# True and initial material models (identical to run_fwi_large_domain.jl)
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
# Generate observed data (once, shared across all lambda values)
# ══════════════════════════════════════════════════════════════════════
println("\nGenerating observed data for $nsrc sources...")
flush(stdout)
obs_datas = Matrix{Float64}[]
for k in 1:nsrc
    t0 = time()
    od = run_forward!(configs[k], eps_inf_true, deps_true, tau_true, sigma_true, src_waveforms[k])
    dt_sec = time() - t0
    push!(obs_datas, od)
    @printf("  Source %d: time = %.1f s\n", k, dt_sec)
    flush(stdout)
end

# Inversion region (identical to run_fwi_large_domain.jl)
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
# Lambda sweep: 5 iterations each
# ══════════════════════════════════════════════════════════════════════
lambda_values = [0.1, 1.0, 10.0, 100.0]

println("\n" * "=" ^ 60)
println("Lambda sweep: 5 iterations each")
println("=" ^ 60)
flush(stdout)

for lam in lambda_values
    println("\n--- lambda = $lam ---")
    flush(stdout)

    Random.seed!(42)  # Reset seed for reproducibility

    t0 = time()
    result = run_fwi_multisource(configs, obs_datas, src_waveforms,
                                  eps_inf_init, deps_init, tau_init, sigma_init,
                                  param_mask;
                                  max_iter=5, param_type=:eps_inf,
                                  use_ad=true, verbose=true,
                                  lower_bound=1.0, upper_bound=25.0,
                                  lambda=lam)
    dt = time() - t0

    # Compute data misfit and regularization separately at final iterate
    # (The reported loss includes both)
    idx_map = GPRADFWI._build_idx_map(param_mask)

    # Extract final parameters
    x_final = Float64[]
    for j in 1:ny, i in 1:nx
        if param_mask[i, j]
            push!(x_final, result.eps_inf_est[i, j])
        end
    end

    reg_penalty = GPRADFWI.tikhonov_penalty(x_final, idx_map, param_mask)
    reg_term = lam * reg_penalty
    data_misfit = result.loss_history[end] - reg_term

    peak_est = maximum(result.eps_inf_est[inv_x_lo:inv_x_hi, inv_y_lo:inv_y_hi])

    @printf("\n  Summary for lambda = %.1e:\n", lam)
    @printf("    Total loss (iter 5): %.6e\n", result.loss_history[end])
    @printf("    Data misfit:         %.6e\n", data_misfit)
    @printf("    Reg term (lambda*R): %.6e\n", reg_term)
    @printf("    Reg / data ratio:    %.4f%%\n", 100.0 * reg_term / max(data_misfit, 1e-30))
    @printf("    Peak eps_inf est:    %.2f (true: 15.0)\n", peak_est)
    @printf("    Time: %.1f s\n", dt)
    flush(stdout)
end

println("\n=== Lambda sweep complete ===")
flush(stdout)
