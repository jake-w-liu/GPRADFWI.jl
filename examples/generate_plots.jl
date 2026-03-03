# generate_plots.jl — Generate heuristic validation plots
#
# Reads CSV data produced by validate_forward.jl and validate_gradients.jl,
# then generates publication-quality figures using PlotlySupply.jl.
#
# Figures:
#   1. heuristic_bscan.pdf — B-scan (functionality check)
#   2. heuristic_gradient_comparison.pdf — AD vs FD gradients (sanity check)
#   3. heuristic_convergence.pdf — FWI convergence (stability check)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "PlotlySupply.jl"))

using PlotlySupply
using PlotlyKaleido: PlotlyKaleido
import PlotlyKaleido: savefig
using DelimitedFiles

PlotlyKaleido.start(mathjax=false)

# IEEE figure constants
const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
const DASHES = ["solid", "dash", "dashdot", "dot"]
const IEEE_SINGLE_COL_W = 504   # 3.5in at 144 DPI
const IEEE_SINGLE_COL_H = 360   # ~2.5in at 144 DPI

datadir = joinpath(@__DIR__, "..", "..", "paper", "data")
figdir  = joinpath(@__DIR__, "..", "..", "paper", "figs")
mkpath(figdir)

# ══════════════════════════════════════════════════════════════════════
# Figure 1: B-scan (receiver gather) — Functionality check
# ══════════════════════════════════════════════════════════════════════
println("Generating heuristic_bscan.pdf ...")

bscan_file = joinpath(datadir, "validation_bscan.csv")
if isfile(bscan_file)
    # Read B-scan: first column = time_ns, remaining = receiver traces
    raw = readdlm(bscan_file, ','; comments=true, comment_char='#')
    # Skip header row if string
    if raw[1, 1] isa AbstractString
        raw = raw[2:end, :]
    end
    time_ns = Float64.(raw[:, 1])
    nrx = size(raw, 2) - 1

    # Select 5 representative traces (evenly spaced)
    trace_ids = round.(Int, range(1, nrx, length=5))

    fig = plot_scatter(time_ns, Float64.(raw[:, trace_ids[1]+1]);
        xlabel="Time [ns]", ylabel="Ez [V/m]",
        title="GPR B-scan: Selected Traces",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="Rx $(trace_ids[1])", linewidth=2)

    for (k, tid) in enumerate(trace_ids[2:end])
        ci = mod(k, 4) + 1
        plot_scatter!(fig, time_ns, Float64.(raw[:, tid+1]);
            color=COLORS[ci], dash=DASHES[ci], mode="lines",
            legend="Rx $tid", linewidth=2)
    end

    set_legend!(fig; position=:topright)
    savefig(fig, joinpath(figdir, "heuristic_bscan.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved heuristic_bscan.pdf")
else
    @warn "B-scan data not found at $bscan_file — skipping plot"
end

# ══════════════════════════════════════════════════════════════════════
# Figure 2: Gradient comparison — Sanity check
# ══════════════════════════════════════════════════════════════════════
println("Generating heuristic_gradient_comparison.pdf ...")

grad_file = joinpath(datadir, "gradient_comparison.csv")
if isfile(grad_file)
    raw = readdlm(grad_file, ','; comments=true, comment_char='#')
    if raw[1, 1] isa AbstractString
        raw = raw[2:end, :]
    end
    param_idx = Float64.(raw[:, 1])
    grad_fd   = Float64.(raw[:, 2])
    grad_ad   = Float64.(raw[:, 3])

    fig = plot_scatter(grad_fd, grad_ad;
        xlabel="FD Gradient", ylabel="AD Gradient",
        title="Gradient Verification: AD vs FD",
        mode="markers", color=COLORS[1],
        marker_size=5, marker_symbol="circle",
        legend="AD vs FD")

    # Add y=x reference line
    gmin = min(minimum(grad_fd), minimum(grad_ad))
    gmax = max(maximum(grad_fd), maximum(grad_ad))
    margin = 0.1 * (gmax - gmin)
    plot_scatter!(fig, [gmin - margin, gmax + margin], [gmin - margin, gmax + margin];
        color=COLORS[2], dash="dash", mode="lines",
        legend="y = x", linewidth=2)

    set_legend!(fig; position=:topleft)
    savefig(fig, joinpath(figdir, "heuristic_gradient_comparison.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved heuristic_gradient_comparison.pdf")
else
    @warn "Gradient data not found at $grad_file — skipping plot"
end

# ══════════════════════════════════════════════════════════════════════
# Figure 3: FWI convergence — Stability check
# ══════════════════════════════════════════════════════════════════════
println("Generating heuristic_convergence.pdf ...")

conv_file = joinpath(datadir, "inversion_convergence.csv")
if isfile(conv_file)
    raw = readdlm(conv_file, ','; comments=true, comment_char='#')
    if raw[1, 1] isa AbstractString
        raw = raw[2:end, :]
    end
    iters     = Float64.(raw[:, 1])
    loss_vals = Float64.(raw[:, 2])
    grad_norm = Float64.(raw[:, 3])

    # Normalize loss
    loss_norm = loss_vals ./ loss_vals[1]

    # Plot both loss and gradient norm on same axes (dual purpose)
    fig = plot_scatter(iters, loss_norm;
        xlabel="Iteration", ylabel="Normalized Value",
        title="FWI Convergence",
        mode="lines+markers", color=COLORS[1], dash=DASHES[1],
        legend="Loss / Loss₀", linewidth=2, marker_size=4)

    grad_norm_normalized = grad_norm ./ grad_norm[1]
    plot_scatter!(fig, iters, grad_norm_normalized;
        mode="lines+markers", color=COLORS[2], dash=DASHES[2],
        legend="||∇f|| / ||∇f₀||", linewidth=2, marker_size=4)

    set_legend!(fig; position=:topright)
    savefig(fig, joinpath(figdir, "heuristic_convergence.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved heuristic_convergence.pdf")
else
    @warn "Convergence data not found at $conv_file — skipping plot"
end

println("\n=== All plots generated ===")
