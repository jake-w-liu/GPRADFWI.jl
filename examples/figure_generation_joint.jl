# figure_generation_joint.jl — Generate figures for joint ε∞ + σ FWI results
#
# Figures:
#   1. fig_results_joint_convergence.pdf          — Convergence history
#   2. fig_results_joint_reconstruction_1d.pdf    — 1D reconstruction (eps + sigma subplots)
#   3. fig_results_joint_reconstruction_2d_eps_*.pdf  — 2D eps_inf maps
#   4. fig_results_joint_reconstruction_2d_sigma_*.pdf — 2D sigma maps

using PlotlySupply
import PlotlySupply: savefig
using DelimitedFiles
using Printf

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
const DASHES = ["solid", "dash", "dashdot", "dot"]
const IEEE_SINGLE_COL_W = 504
const IEEE_SINGLE_COL_H = 360
const IEEE_DOUBLE_COL_W = 1008

datadir = joinpath(@__DIR__, "..", "..", "paper", "data")
figdir  = joinpath(@__DIR__, "..", "..", "paper", "figs")
mkpath(figdir)

function read_csv(filepath)
    raw = readdlm(filepath, ','; comments=true, comment_char='#')
    if raw[1, 1] isa AbstractString
        raw = raw[2:end, :]
    end
    return raw
end

function read_2d_csv(filepath)
    raw = readdlm(filepath, ','; comments=true, comment_char='#')
    if raw[1, 1] isa AbstractString
        raw = raw[2:end, :]
    end
    x_all = Float64.(raw[:, 1])
    y_all = Float64.(raw[:, 2])
    v_all = Float64.(raw[:, 3])

    x_vec = sort(unique(x_all))
    y_vec = sort(unique(y_all))
    nx_u = length(x_vec)
    ny_u = length(y_vec)

    U = zeros(nx_u, ny_u)
    x_idx = Dict(x => i for (i, x) in enumerate(x_vec))
    y_idx = Dict(y => j for (j, y) in enumerate(y_vec))
    for k in 1:length(x_all)
        i = x_idx[x_all[k]]
        j = y_idx[y_all[k]]
        U[i, j] = v_all[k]
    end
    return x_vec, y_vec, U
end

# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Convergence history (joint vs single-parameter overlay)
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_joint_convergence.pdf ...")
conv_joint = joinpath(datadir, "fwi_joint_convergence.csv")
conv_single = joinpath(datadir, "fwi_large_convergence.csv")

if isfile(conv_joint)
    raw_j = read_csv(conv_joint)
    iter_j = Float64.(raw_j[:, 1])
    loss_j = Float64.(raw_j[:, 2])

    fig_conv = plot_scatter(iter_j, loss_j;
        xlabel="Iteration", ylabel="Regularized Objective",
        mode="lines+markers", color=COLORS[2], dash=DASHES[2],
        legend="Joint (eps_inf + sigma)", linewidth=2, marker_size=3,
        yscale="log")

    # Overlay single-parameter convergence if available
    if isfile(conv_single)
        raw_s = read_csv(conv_single)
        iter_s = Float64.(raw_s[:, 1])
        loss_s = Float64.(raw_s[:, 2])
        plot_scatter!(fig_conv, iter_s, loss_s;
            color=COLORS[1], dash=DASHES[1], mode="lines+markers",
            legend="Single (eps_inf only)", linewidth=2, marker_size=3)
    end

    # Legend: topright is least obstructive because the convergence traces fall away from that corner.
    set_legend!(fig_conv; position=:topright)
    savefig(fig_conv, joinpath(figdir, "fig_results_joint_convergence.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_joint_convergence.pdf")
else
    println("  SKIPPED (data not found)")
end

# ══════════════════════════════════════════════════════════════════════════
# Figure 2a: 1D reconstruction — eps_inf
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_joint_reconstruction_1d_eps.pdf ...")
recon1d = joinpath(datadir, "fwi_joint_reconstruction_1d.csv")

if isfile(recon1d)
    raw = read_csv(recon1d)
    depth       = Float64.(raw[:, 1])
    eps_true    = Float64.(raw[:, 2])
    eps_init    = Float64.(raw[:, 3])
    eps_est     = Float64.(raw[:, 4])
    sigma_true  = Float64.(raw[:, 5])
    sigma_init  = Float64.(raw[:, 6])
    sigma_est   = Float64.(raw[:, 7])

    # Panel (a): eps_inf — displayed as subfigure, so use larger font
    fig_eps = plot_scatter(depth, eps_true;
        xlabel="Depth [cm]", ylabel="Relative Permittivity",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="True", linewidth=2, fontsize=22)
    plot_scatter!(fig_eps, depth, eps_init;
        mode="lines", color=COLORS[3], dash=DASHES[3],
        legend="Initial", linewidth=2)
    plot_scatter!(fig_eps, depth, eps_est;
        mode="lines+markers", color=COLORS[2], dash=DASHES[2],
        legend="Joint FWI", linewidth=2, marker_size=3)
    # Legend: topright is least obstructive because profile separation is strongest below the shallow interval.
    set_legend!(fig_eps; position=:topright)
    savefig(fig_eps, joinpath(figdir, "fig_results_joint_reconstruction_1d_eps.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_joint_reconstruction_1d_eps.pdf")

    # Panel (b): sigma
    println("Generating fig_results_joint_reconstruction_1d_sigma.pdf ...")
    # Panel (b): sigma — displayed as subfigure, so use larger font
    fig_sig = plot_scatter(depth, sigma_true .* 1e3;
        xlabel="Depth [cm]", ylabel="Conductivity [mS/m]",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="True", linewidth=2, fontsize=22)
    plot_scatter!(fig_sig, depth, sigma_init .* 1e3;
        mode="lines", color=COLORS[3], dash=DASHES[3],
        legend="Initial", linewidth=2)
    plot_scatter!(fig_sig, depth, sigma_est .* 1e3;
        mode="lines+markers", color=COLORS[2], dash=DASHES[2],
        legend="Joint FWI", linewidth=2, marker_size=3)
    # Legend: topright is least obstructive because conductivity variation is concentrated below the shallow interval.
    set_legend!(fig_sig; position=:topright)
    savefig(fig_sig, joinpath(figdir, "fig_results_joint_reconstruction_1d_sigma.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_joint_reconstruction_1d_sigma.pdf")
else
    println("  SKIPPED (data not found)")
end

# ══════════════════════════════════════════════════════════════════════════
# Figure 3: 2D eps_inf maps (true / initial / estimated)
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_joint_reconstruction_2d_eps.pdf ...")

eps_files = [
    ("true",      joinpath(datadir, "fwi_joint_reconstruction_2d_eps_true.csv")),
    ("initial",   joinpath(datadir, "fwi_joint_reconstruction_2d_eps_initial.csv")),
    ("estimated", joinpath(datadir, "fwi_joint_reconstruction_2d_eps_estimated.csv")),
]

if all(isfile(f) for (_, f) in eps_files)
    # Read all to find shared color range
    all_data = []
    for (tag, fpath) in eps_files
        x, y, U = read_2d_csv(fpath)
        push!(all_data, (tag, x, y, U))
    end
    zmin_e = minimum(minimum(d[4]) for d in all_data) - 0.5
    zmax_e = maximum(maximum(d[4]) for d in all_data) + 0.5

    for (tag, x, y, U) in all_data
        fname = "fig_results_joint_reconstruction_2d_eps_$(tag).pdf"
        fig = plot_heatmap(x, y, U;
            xlabel="x [cm]", ylabel="Depth [cm]",
            colorscale="Viridis", zrange=[zmin_e, zmax_e],
            equalar=true,
            yrange=[maximum(y), minimum(y)])
        savefig(fig, joinpath(figdir, fname); width=336, height=IEEE_SINGLE_COL_H)
        println("  Saved $fname")
    end
else
    println("  SKIPPED (data not found)")
end

# ══════════════════════════════════════════════════════════════════════════
# Figure 4: 2D sigma maps (true / initial / estimated)
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_joint_reconstruction_2d_sigma.pdf ...")

sigma_files = [
    ("true",      joinpath(datadir, "fwi_joint_reconstruction_2d_sigma_true.csv")),
    ("initial",   joinpath(datadir, "fwi_joint_reconstruction_2d_sigma_initial.csv")),
    ("estimated", joinpath(datadir, "fwi_joint_reconstruction_2d_sigma_estimated.csv")),
]

if all(isfile(f) for (_, f) in sigma_files)
    all_data_s = []
    for (tag, fpath) in sigma_files
        x, y, U = read_2d_csv(fpath)
        # Convert to mS/m for display
        push!(all_data_s, (tag, x, y, U .* 1e3))
    end
    zmin_s = max(0.0, minimum(minimum(d[4]) for d in all_data_s) - 1.0)
    zmax_s = maximum(maximum(d[4]) for d in all_data_s) + 1.0

    for (tag, x, y, U) in all_data_s
        fname = "fig_results_joint_reconstruction_2d_sigma_$(tag).pdf"
        fig = plot_heatmap(x, y, U;
            xlabel="x [cm]", ylabel="Depth [cm]",
            colorscale="Viridis", zrange=[zmin_s, zmax_s],
            equalar=true,
            yrange=[maximum(y), minimum(y)])
        savefig(fig, joinpath(figdir, fname); width=336, height=IEEE_SINGLE_COL_H)
        println("  Saved $fname")
    end
else
    println("  SKIPPED (data not found)")
end

println("\n=== Joint inversion figures generated ===")
