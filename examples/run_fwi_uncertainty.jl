# run_fwi_uncertainty.jl
# Reduced-domain uncertainty stress tests for revision item C2.

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

function main()
    Random.seed!(20260303)

    println("Building reduced-domain context for uncertainty experiments...")
    ctx = build_reduced_multisource_context()
    @printf("  Domain: %d x %d, nt=%d\n", ctx.nx, ctx.ny, ctx.configs[1].nt)
    @printf("  Sources: %d, parameters: %d\n", length(ctx.configs), ctx.n_params)
    flush(stdout)

    cases = [
        (id="reference_known_background", label="Reference (known layer boundary, nominal Debye)", layer_shift_cells=0, deps_scale=1.0, tau_scale=1.0),
        (id="boundary_shift_plus4cm", label="Boundary-depth uncertainty (+4 cm)", layer_shift_cells=4, deps_scale=1.0, tau_scale=1.0),
        (id="debye_perturbed", label="Fixed Debye mismatch (deps x0.8, tau x1.25)", layer_shift_cells=0, deps_scale=0.8, tau_scale=1.25),
    ]

    summary_rows = NamedTuple[]

    for case in cases
        println("\n" * "="^70)
        println("Case: $(case.label)")
        println("="^70)
        flush(stdout)

        eps_init, deps_init, tau_init, sigma_init =
            build_initial_model(ctx; layer_shift_cells=case.layer_shift_cells)

        deps_fixed = case.deps_scale .* deps_init
        tau_fixed = case.tau_scale .* tau_init

        t0 = time()
        result = run_fwi_multisource(
            ctx.configs,
            ctx.obs_datas_clean,
            ctx.src_waveforms,
            eps_init,
            deps_fixed,
            tau_fixed,
            sigma_init,
            ctx.param_mask;
            max_iter=15,
            param_type=:eps_inf,
            use_ad=true,
            verbose=true,
            lower_bound=1.0,
            upper_bound=25.0,
            lambda=1.0,
        )
        wall_s = time() - t0

        rmse, peak_true, peak_est, recovery = compute_region_metrics(ctx, result.eps_inf_est)
        reduction = 100.0 * (1.0 - result.loss_history[end] / result.loss_history[1])

        @printf("  Runtime: %.1f s\n", wall_s)
        @printf("  Loss: %.4e -> %.4e (%.2f%% reduction)\n",
                result.loss_history[1], result.loss_history[end], reduction)
        @printf("  RMSE: %.4f, peak %.2f / %.2f (%.1f%%)\n",
                rmse, peak_est, peak_true, recovery)
        flush(stdout)

        conv_file = joinpath(DATADIR, "fwi_uncertainty_$(case.id)_convergence.csv")
        save_convergence_csv(conv_file, result;
            header="Uncertainty case $(case.id), max_iter=15, nsrc=$(length(ctx.configs))")

        recon_file = joinpath(DATADIR, "fwi_uncertainty_$(case.id)_reconstruction_1d.csv")
        save_reconstruction_slice_csv(recon_file, ctx, result.eps_inf_est, eps_init;
            header="Uncertainty case $(case.id), centerline reconstruction")

        push!(summary_rows, (
            case_id=case.id,
            layer_shift_cm=case.layer_shift_cells * ctx.grid_dx * 100.0,
            deps_scale=case.deps_scale,
            tau_scale=case.tau_scale,
            runtime_s=wall_s,
            loss_initial=result.loss_history[1],
            loss_final=result.loss_history[end],
            loss_reduction_pct=reduction,
            rmse=rmse,
            peak_true=peak_true,
            peak_est=peak_est,
            peak_recovery_pct=recovery,
        ))

        println("  Saved: $conv_file")
        println("  Saved: $recon_file")
    end

    summary_file = joinpath(DATADIR, "fwi_uncertainty_summary.csv")
    open(summary_file, "w") do io
        write(io, "# Reduced-domain uncertainty stress test summary\n")
        write(io, "case_id,layer_shift_cm,deps_scale,tau_scale,runtime_s,loss_initial,loss_final,loss_reduction_pct,rmse,peak_true,peak_est,peak_recovery_pct\n")
        for row in summary_rows
            @printf(io, "%s,%.2f,%.4f,%.4f,%.3f,%.12e,%.12e,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    row.case_id,
                    row.layer_shift_cm,
                    row.deps_scale,
                    row.tau_scale,
                    row.runtime_s,
                    row.loss_initial,
                    row.loss_final,
                    row.loss_reduction_pct,
                    row.rmse,
                    row.peak_true,
                    row.peak_est,
                    row.peak_recovery_pct)
        end
    end

    println("\nSaved: $summary_file")
    println("\n=== Uncertainty stress tests complete ===")
end

main()
