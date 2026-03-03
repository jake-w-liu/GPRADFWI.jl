# benchmark_timing.jl — Time forward solve vs AD gradient vs FD gradient
#
# Measures wall-clock times on the gradient verification domain (60×50, 300 steps)
# to quantify the AD gradient overhead relative to one forward solve.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using GPRADFWI
using Enzyme
using Printf
using Random
using LinearAlgebra
using Statistics

# Direct Enzyme.autodiff on forward_misfit (not closures — Enzyme
# requires named functions for mutability analysis).
function enzyme_gradient(x0, config, obs_data, src_wf, eps_inf_bg,
                          deps_true, tau_true, sigma_bg, param_mask)
    dx = zeros(length(x0))
    Enzyme.autodiff(Enzyme.Reverse, forward_misfit, Enzyme.Active,
        Enzyme.Duplicated(x0, dx),
        Enzyme.Const(config),
        Enzyme.Const(obs_data),
        Enzyme.Const(src_wf),
        Enzyme.Const(eps_inf_bg),
        Enzyme.Const(deps_true),
        Enzyme.Const(tau_true),
        Enzyme.Const(sigma_bg),
        Enzyme.Const(param_mask),
        Enzyme.Const(:eps_inf))
    return dx
end

function main()
    Random.seed!(42)
    datadir = joinpath(@__DIR__, "..", "..", "paper", "data")
    mkpath(datadir)

    # ── Domain setup (same as validate_gradients.jl) ──────────────────
    nx = 60
    ny = 50
    dx_grid = 0.01
    fc = 300e6
    npml = 8

    rx_list = collect((npml+2):3:(nx-npml-2))
    rx_y = npml + 5
    src_x = nx ÷ 2
    src_y = rx_y

    config = create_config(
        nx=nx, ny=ny, dx=dx_grid, fc=fc, npml=npml,
        src_ix=src_x, src_iy=src_y,
        rx_iy=rx_y, rx_ix_list=rx_list,
        nt=300,
    )

    # True model with anomaly
    eps_inf_true = 3.0 * ones(nx, ny)
    deps_true = zeros(nx, ny)
    tau_true = zeros(nx, ny)
    sigma_true = 0.001 * ones(nx, ny)

    anomaly_i = nx ÷ 2
    anomaly_j = ny ÷ 2
    anomaly_r = 3
    for j in 1:ny, i in 1:nx
        if (i - anomaly_i)^2 + (j - anomaly_j)^2 <= anomaly_r^2
            eps_inf_true[i, j] = 6.0
        end
    end

    src_wf = create_source(config)
    obs_data = run_forward!(config, eps_inf_true, deps_true, tau_true, sigma_true, src_wf)

    # Inversion mask
    param_mask = falses(nx, ny)
    inv_region = 6
    for j in (anomaly_j-inv_region):(anomaly_j+inv_region)
        for i in (anomaly_i-inv_region):(anomaly_i+inv_region)
            if 1 <= i <= nx && 1 <= j <= ny
                param_mask[i, j] = true
            end
        end
    end
    n_params = count(param_mask)

    eps_inf_bg = 3.0 * ones(nx, ny)
    sigma_bg = 0.001 * ones(nx, ny)

    function obj_eps(x_flat)
        return forward_misfit(x_flat, config, obs_data, src_wf,
                              eps_inf_bg, deps_true, tau_true, sigma_bg,
                              param_mask, :eps_inf)
    end

    x0 = Float64[]
    for j in 1:ny, i in 1:nx
        if param_mask[i, j]
            push!(x0, eps_inf_bg[i, j])
        end
    end

    println("Domain: $(nx)×$(ny), $(config.nt) time steps, $n_params inversion params")
    println()

    # ── Benchmark forward solve ──────────────────────────────────────
    println("Benchmarking forward solve...")
    # Warmup
    run_forward!(config, eps_inf_bg, deps_true, tau_true, sigma_bg, src_wf)
    # Timed runs
    n_fwd = 10
    t_fwd_samples = Float64[]
    t_fwd_total = 0.0
    for _ in 1:n_fwd
        t = time()
        run_forward!(config, eps_inf_bg, deps_true, tau_true, sigma_bg, src_wf)
        dt = time() - t
        t_fwd_total += dt
        push!(t_fwd_samples, dt)
    end
    t_fwd = t_fwd_total / n_fwd
    @printf("  Forward solve: %.4f s (avg of %d runs)\n", t_fwd, n_fwd)
    flush(stdout)

    # ── Benchmark objective evaluation ────────────────────────────────
    println("Benchmarking objective evaluation (forward + misfit)...")
    obj_eps(x0)  # warmup
    n_obj = 10
    t_obj_samples = Float64[]
    t_obj_total = 0.0
    for _ in 1:n_obj
        t = time()
        obj_eps(x0)
        dt = time() - t
        t_obj_total += dt
        push!(t_obj_samples, dt)
    end
    t_obj = t_obj_total / n_obj
    @printf("  Objective eval: %.4f s (avg of %d runs)\n", t_obj, n_obj)
    flush(stdout)

    # ── Benchmark AD gradient (Enzyme) ────────────────────────────────
    println("Benchmarking Enzyme AD gradient (first call = compilation)...")
    flush(stdout)
    t_ad_first = time()
    grad_ad = enzyme_gradient(x0, config, obs_data, src_wf, eps_inf_bg,
                               deps_true, tau_true, sigma_bg, param_mask)
    t_ad_first = time() - t_ad_first
    @printf("  AD gradient (1st call, incl. compilation): %.2f s\n", t_ad_first)
    flush(stdout)

    # Subsequent calls
    n_ad = 5
    t_ad_samples = Float64[]
    t_ad_total = 0.0
    for run in 1:n_ad
        t = time()
        grad_ad = enzyme_gradient(x0, config, obs_data, src_wf, eps_inf_bg,
                                   deps_true, tau_true, sigma_bg, param_mask)
        dt = time() - t
        t_ad_total += dt
        push!(t_ad_samples, dt)
        @printf("  AD gradient (run %d): %.4f s\n", run+1, dt)
        flush(stdout)
    end
    t_ad = t_ad_total / n_ad
    @printf("  AD gradient (steady-state avg): %.4f s\n", t_ad)
    flush(stdout)

    # ── Benchmark FD gradient ──────────────────────────────────────────
    println("Benchmarking FD gradient...")
    flush(stdout)
    t_fd_start = time()
    grad_fd = fd_gradient(obj_eps, x0; h=1e-5)
    t_fd = time() - t_fd_start
    @printf("  FD gradient: %.2f s (%d params × 2 evals = %d forward solves)\n",
            t_fd, n_params, 2*n_params)
    flush(stdout)

    # ── Summary ────────────────────────────────────────────────────────
    println()
    println("=" ^ 60)
    println("TIMING SUMMARY")
    println("=" ^ 60)
    @printf("  Forward solve:              %.4f s\n", t_fwd)
    @printf("  Objective eval:             %.4f s\n", t_obj)
    @printf("  AD gradient (steady-state): %.4f s\n", t_ad)
    @printf("  AD gradient / forward:      %.1f×\n", t_ad / t_fwd)
    @printf("  FD gradient:                %.2f s\n", t_fd)
    @printf("  FD gradient / forward:      %.0f× (%d evals)\n", t_fd / t_fwd, 2*n_params)
    @printf("  AD speedup vs FD:           %.0f×\n", t_fd / t_ad)
    println()
    @printf("  Hand-adjoint estimate:      ≈ 2× forward = %.4f s\n", 2*t_fwd)
    @printf("  AD overhead vs hand-adjoint: %.1f×\n", t_ad / (2*t_fwd))
    println()
    println("Note: AD first-call includes Enzyme compilation (one-time cost).")
    @printf("  AD first call: %.2f s\n", t_ad_first)
    flush(stdout)

    # ── Archive reproducibility artifacts ──────────────────────────────
    env_file = joinpath(datadir, "timing_environment.csv")
    cpu_model = Sys.cpu_info()[1].model
    open(env_file, "w") do io
        write(io, "# Environment metadata for benchmark_timing.jl\n")
        write(io, "key,value\n")
        @printf(io, "julia_version,%s\n", string(VERSION))
        @printf(io, "threads,%d\n", Threads.nthreads())
        @printf(io, "cpu_threads,%d\n", Sys.CPU_THREADS)
        @printf(io, "word_size_bits,%d\n", Sys.WORD_SIZE)
        @printf(io, "kernel,%s\n", Sys.KERNEL)
        @printf(io, "machine,%s\n", Sys.MACHINE)
        @printf(io, "cpu_model,%s\n", replace(cpu_model, "," => ";"))
        @printf(io, "domain_nx,%d\n", nx)
        @printf(io, "domain_ny,%d\n", ny)
        @printf(io, "time_steps,%d\n", config.nt)
        @printf(io, "n_params,%d\n", n_params)
        @printf(io, "n_fwd_runs,%d\n", n_fwd)
        @printf(io, "n_obj_runs,%d\n", n_obj)
        @printf(io, "n_ad_runs,%d\n", n_ad)
    end

    sample_file = joinpath(datadir, "timing_benchmark_samples.csv")
    open(sample_file, "w") do io
        write(io, "# Raw timing samples for benchmark_timing.jl\n")
        write(io, "operation,run_id,time_s\n")
        for (k, t) in enumerate(t_fwd_samples)
            @printf(io, "forward,%d,%.12e\n", k, t)
        end
        for (k, t) in enumerate(t_obj_samples)
            @printf(io, "objective,%d,%.12e\n", k, t)
        end
        @printf(io, "ad_gradient_first_call,1,%.12e\n", t_ad_first)
        for (k, t) in enumerate(t_ad_samples)
            @printf(io, "ad_gradient,%d,%.12e\n", k, t)
        end
        @printf(io, "fd_gradient,1,%.12e\n", t_fd)
    end

    summary_file = joinpath(datadir, "timing_benchmark_summary.csv")
    open(summary_file, "w") do io
        write(io, "# Timing summary statistics for benchmark_timing.jl\n")
        write(io, "operation,mean_s,std_s,min_s,max_s,relative_to_forward\n")
        @printf(io, "forward,%.12e,%.12e,%.12e,%.12e,1.0\n",
                mean(t_fwd_samples), std(t_fwd_samples), minimum(t_fwd_samples), maximum(t_fwd_samples))
        @printf(io, "objective,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                mean(t_obj_samples), std(t_obj_samples), minimum(t_obj_samples), maximum(t_obj_samples),
                mean(t_obj_samples) / mean(t_fwd_samples))
        @printf(io, "ad_gradient_steady,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                mean(t_ad_samples), std(t_ad_samples), minimum(t_ad_samples), maximum(t_ad_samples),
                mean(t_ad_samples) / mean(t_fwd_samples))
        @printf(io, "ad_gradient_first_call,%.12e,0.0,%.12e,%.12e,%.12e\n",
                t_ad_first, t_ad_first, t_ad_first, t_ad_first / mean(t_fwd_samples))
        @printf(io, "fd_gradient,%.12e,0.0,%.12e,%.12e,%.12e\n",
                t_fd, t_fd, t_fd, t_fd / mean(t_fwd_samples))
        @printf(io, "hand_adjoint_theory_2x_forward,%.12e,0.0,%.12e,%.12e,2.0\n",
                2 * mean(t_fwd_samples), 2 * mean(t_fwd_samples), 2 * mean(t_fwd_samples))
    end

    println()
    println("Archived benchmark artifacts:")
    println("  $env_file")
    println("  $sample_file")
    println("  $summary_file")
end

main()
