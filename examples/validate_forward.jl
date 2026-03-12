# validate_forward.jl — Forward solver validation and heuristic checks
#
# Generates:
#   1. B-scan (receiver gather) for a layered soil model
#   2. Wavefield snapshot showing propagation
#   3. CSV data files for all outputs

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using GPRADFWI
using DelimitedFiles
using Printf
using Random

Random.seed!(42)

# ── Problem setup: 2-layer soil with buried pipe ──────────────────────
# Domain: 2m wide × 1.5m deep
# Grid: dx = dy = 5mm (fine enough for 1 GHz GPR)
const domain_x = 2.0    # [m]
const domain_y = 1.5    # [m]
const grid_dx  = 0.005  # [m] (5 mm)
const fc_gpr   = 500e6  # [Hz] center frequency (500 MHz GPR)

const nx = round(Int, domain_x / grid_dx)   # 400
const ny = round(Int, domain_y / grid_dx)    # 300
const npml = 15

# Receiver line at surface (y ≈ 5cm below top)
const rx_y = npml + 10  # just below air-surface interface
const rx_x_list = collect((npml+5):4:(nx-npml-5))  # every 2cm across

# Source at center of receiver line
const src_x = nx ÷ 2
const src_y = rx_y

println("Domain: $(nx) × $(ny) cells, dx = $(grid_dx*1e3) mm")
println("Receivers: $(length(rx_x_list)) at y-index $rx_y")
println("Source at ($src_x, $src_y)")

config = create_config(
    nx=nx, ny=ny, dx=grid_dx, fc=fc_gpr, npml=npml,
    src_ix=src_x, src_iy=src_y,
    rx_iy=rx_y, rx_ix_list=rx_x_list,
)
println("Time steps: $(config.nt), dt = $(@sprintf("%.4e", config.dt)) s")

# ── Build material model ──────────────────────────────────────────────
# Air: eps_inf=1, no dispersion
# Soil layer 1 (0.2m - 0.8m depth): sandy soil with Debye dispersion
# Soil layer 2 (0.8m - bottom): clay with stronger dispersion
# Buried pipe: high permittivity anomaly at (1.0m, 0.5m), radius 5cm

eps_inf_map = ones(nx, ny)
deps_map    = zeros(nx, ny)
tau_map     = zeros(nx, ny)
sigma_map   = zeros(nx, ny)

# Surface at y-index corresponding to ~0.1m from top
surface_j = npml + 20  # ~ 0.1m from domain top

# Layer 1: sandy soil (Debye: ε∞=4.0, Δε=4.0, τ=0.3ns, σ=0.005 S/m)
layer1_top = surface_j
layer1_bot = surface_j + round(Int, 0.6 / grid_dx)

for j in layer1_top:min(layer1_bot, ny), i in 1:nx
    eps_inf_map[i, j] = 4.0
    deps_map[i, j]    = 4.0
    tau_map[i, j]     = 0.3e-9   # 0.3 ns
    sigma_map[i, j]   = 0.005
end

# Layer 2: clay (Debye: ε∞=6.0, Δε=12.0, τ=1.0ns, σ=0.02 S/m)
layer2_top = layer1_bot + 1

for j in layer2_top:ny, i in 1:nx
    eps_inf_map[i, j] = 6.0
    deps_map[i, j]    = 12.0
    tau_map[i, j]     = 1.0e-9
    sigma_map[i, j]   = 0.02
end

# Buried pipe: cylinder at (1.0m, 0.5m below surface), radius 5cm
pipe_cx = round(Int, 1.0 / grid_dx)   # center x
pipe_cy = surface_j + round(Int, 0.5 / grid_dx)  # center y
pipe_r  = round(Int, 0.05 / grid_dx)  # radius in cells

for j in 1:ny, i in 1:nx
    r2 = (i - pipe_cx)^2 + (j - pipe_cy)^2
    if r2 <= pipe_r^2
        eps_inf_map[i, j] = 15.0   # high permittivity (water-filled pipe)
        deps_map[i, j]    = 10.0
        tau_map[i, j]     = 0.5e-9
        sigma_map[i, j]   = 0.001
    end
end

println("\nMaterial model built:")
println("  Layer 1 (sandy soil): j=$(layer1_top):$(layer1_bot), ε∞=4, Δε=4, τ=0.3ns")
println("  Layer 2 (clay): j=$(layer2_top):$(ny), ε∞=6, Δε=12, τ=1.0ns")
println("  Pipe: center=($pipe_cx,$pipe_cy), r=$pipe_r cells")

# ── Run forward simulation ────────────────────────────────────────────
src_waveform = create_source(config)

println("\nRunning forward simulation...")
t_start = time()
rec_data = run_forward!(config, eps_inf_map, deps_map, tau_map, sigma_map, src_waveform)
t_elapsed = time() - t_start
@printf("Forward simulation done in %.2f s\n", t_elapsed)

# ── Sanity checks ─────────────────────────────────────────────────────
println("\n=== Sanity Checks ===")
max_ez = maximum(abs.(rec_data))
any_nan = any(isnan.(rec_data))
any_inf = any(isinf.(rec_data))
println("  Max |Ez| at receivers: $(@sprintf("%.6e", max_ez))")
println("  Any NaN: $any_nan")
println("  Any Inf: $any_inf")
@assert !any_nan "NaN detected in receiver data!"
@assert !any_inf "Inf detected in receiver data!"
@assert max_ez > 0 "Zero signal at receivers!"
@assert max_ez < 1e10 "Signal magnitude unreasonably large (>1e10)!"

# ── Save CSV data ─────────────────────────────────────────────────────
datadir = joinpath(@__DIR__, "..", "..", "paper", "data")
mkpath(datadir)

# B-scan data
bscan_file = joinpath(datadir, "validation_bscan.csv")
open(bscan_file, "w") do io
    # Header
    write(io, "# Forward simulation B-scan: $(length(rx_x_list)) receivers, $(config.nt) time steps, seed=42\n")
    write(io, "time_ns")
    for r in 1:length(rx_x_list)
        write(io, ",rx$(r)_ez_vm")
    end
    write(io, "\n")
    # Data
    for n in 1:config.nt
        t_ns = n * config.dt * 1e9
        @printf(io, "%.6e", t_ns)
        for r in 1:length(rx_x_list)
            @printf(io, ",%.6e", rec_data[n, r])
        end
        write(io, "\n")
    end
end
println("\nB-scan saved to: $bscan_file")

# Source waveform
src_file = joinpath(datadir, "validation_source.csv")
open(src_file, "w") do io
    write(io, "# Ricker wavelet, fc=$(fc_gpr/1e6) MHz, t0=$(config.source.t0*1e9) ns\n")
    write(io, "time_ns,amplitude\n")
    for n in 1:config.nt
        t_ns = n * config.dt * 1e9
        @printf(io, "%.6e,%.6e\n", t_ns, src_waveform[n])
    end
end
println("Source waveform saved to: $src_file")

# Material profile (vertical slice at center)
mat_file = joinpath(datadir, "validation_material_profile.csv")
open(mat_file, "w") do io
    write(io, "# Material profile at x=$(pipe_cx*grid_dx) m\n")
    write(io, "depth_m,eps_inf,delta_eps,tau_ns,sigma_sm\n")
    for j in 1:ny
        depth = (j - surface_j) * grid_dx
        @printf(io, "%.6e,%.6e,%.6e,%.6e,%.6e\n",
                depth, eps_inf_map[pipe_cx, j], deps_map[pipe_cx, j],
                tau_map[pipe_cx, j] * 1e9, sigma_map[pipe_cx, j])
    end
end
println("Material profile saved to: $mat_file")

# Combined validation summary CSV
summary_file = joinpath(datadir, "validation_data.csv")
open(summary_file, "w") do io
    write(io, "# Validation summary: forward solver, seed=42\n")
    write(io, "metric_name,value_si,unit\n")
    @printf(io, "domain_x_m,%.4f,m\n", domain_x)
    @printf(io, "domain_y_m,%.4f,m\n", domain_y)
    @printf(io, "grid_dx_mm,%.2f,mm\n", grid_dx * 1e3)
    @printf(io, "nx,%d,cells\n", nx)
    @printf(io, "ny,%d,cells\n", ny)
    @printf(io, "nt,%d,steps\n", config.nt)
    @printf(io, "dt_ps,%.4f,ps\n", config.dt * 1e12)
    @printf(io, "fc_mhz,%.1f,MHz\n", fc_gpr / 1e6)
    @printf(io, "n_receivers,%d,count\n", length(rx_x_list))
    @printf(io, "max_ez_vm,%.6e,V/m\n", max_ez)
    @printf(io, "runtime_s,%.2f,s\n", t_elapsed)
    @printf(io, "has_missing_flag,%s,bool\n", any_nan)
    @printf(io, "has_inf_flag,%s,bool\n", any_inf)
end
println("Summary saved to: $summary_file")

println("\n=== Forward validation complete ===")
