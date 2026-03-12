# summarize_full_domain_uncertainty.jl
# Build a full-domain imperfect-prior summary table from archived convergence CSVs.

using DelimitedFiles
using Printf

const DATADIR = joinpath(@__DIR__, "..", "..", "paper", "data")

function read_convergence_metrics(path::String)
    @assert isfile(path) "Missing convergence file: $path"
    raw = readdlm(path, ','; comments=true, comment_char='#')
    if raw[1, 1] isa AbstractString
        raw = raw[2:end, :]
    end
    iter = Int.(raw[:, 1])
    loss = Float64.(raw[:, 2])  # loss
    return (
        n_iter=iter[end],
        loss_initial=loss[1],
        loss_final=loss[end],
        loss_reduction_pct=100.0 * (1.0 - loss[end] / loss[1]),
    )
end

function main()
    ref = read_convergence_metrics(joinpath(DATADIR, "fwi_large_convergence.csv"))
    nondisp = read_convergence_metrics(joinpath(DATADIR, "fwi_nondispersive_convergence.csv"))

    summary_path = joinpath(DATADIR, "fwi_uncertainty_full_domain_summary.csv")
    open(summary_path, "w") do io
        write(io, "# Full-domain imperfect-prior stress summary from archived convergence traces\n")
        write(io, "case_id,model_assumption,n_iter,loss_initial,loss_final,loss_reduction_pct,initial_loss_ratio_vs_reference\n")
        @printf(io, "%s,%s,%d,%.12e,%.12e,%.6f,%.6f\n",
                "reference_dispersive",
                "Debye priors matched (baseline)",
                ref.n_iter,
                ref.loss_initial,
                ref.loss_final,
                ref.loss_reduction_pct,
                1.0)
        @printf(io, "%s,%s,%d,%.12e,%.12e,%.6f,%.6f\n",
                "imperfect_prior_nondispersive",
                "Debye priors misspecified: deps=0; tau=0",
                nondisp.n_iter,
                nondisp.loss_initial,
                nondisp.loss_final,
                nondisp.loss_reduction_pct,
                nondisp.loss_initial / ref.loss_initial)
    end

    println("Saved: $summary_path")
end

main()
