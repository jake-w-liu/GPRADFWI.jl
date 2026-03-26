# figure_generation_m1_m3.jl — Generate figures for M1 and M3 revision experiments
#
# M1 figures (three-way dispersive comparison):
#   1. fig_results_dispersive_comparison_convergence.pdf — Updated with ε_s curve
#
# M3 figures (two-anomaly model):
#   2. fig_results_two_anomaly_convergence.pdf
#   3. fig_results_two_anomaly_reconstruction_2d.pdf (true, initial, estimated)
#   4. fig_results_two_anomaly_reconstruction_1d.pdf (pipe + void slices)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "PlotlySupply.jl"))

using PlotlySupply
using PlotlyKaleido: PlotlyKaleido
import PlotlyKaleido: savefig
using DelimitedFiles
using Printf

try
    PlotlyKaleido.start(mathjax=false, timeout=30)
catch
    PlotlyKaleido.restart(mathjax=false, timeout=60)
end

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
const DASHES = ["solid", "dash", "dashdot", "dot"]
const IEEE_SINGLE_COL_W = 504
const IEEE_SINGLE_COL_H = 360

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
# M1: Updated three-way convergence comparison
# ══════════════════════════════════════════════════════════════════════════
println("=== M1: Three-way dispersive comparison ===")

conv_disp = joinpath(datadir, "fwi_large_convergence.csv")
conv_nondisp = joinpath(datadir, "fwi_nondispersive_convergence.csv")
conv_eps_s = joinpath(datadir, "fwi_nondispersive_eps_s_convergence.csv")

if isfile(conv_disp) && isfile(conv_nondisp) && isfile(conv_eps_s)
    raw_d = read_csv(conv_disp)
    raw_n = read_csv(conv_nondisp)
    raw_s = read_csv(conv_eps_s)

    iter_d = Float64.(raw_d[:, 1])
    loss_d = Float64.(raw_d[:, 2])
    iter_n = Float64.(raw_n[:, 1])
    loss_n = Float64.(raw_n[:, 2])
    iter_s = Float64.(raw_s[:, 1])
    loss_s = Float64.(raw_s[:, 2])

    fig_cc = plot_scatter(iter_d, loss_d;
        xlabel="Iteration", ylabel="Regularized Objective",
        mode="lines+markers", color=COLORS[1], dash=DASHES[1],
        legend="Dispersive FWI", linewidth=2, marker_size=4,
        yscale="log")

    plot_scatter!(fig_cc, iter_s, loss_s;
        color=COLORS[3], dash=DASHES[3], mode="lines+markers",
        legend=raw"Non-dispersive ($\varepsilon_s$)", linewidth=2, marker_size=4)

    plot_scatter!(fig_cc, iter_n, loss_n;
        color=COLORS[2], dash=DASHES[2], mode="lines+markers",
        legend=raw"Non-dispersive ($\varepsilon_\infty$)", linewidth=2, marker_size=4)

    set_legend!(fig_cc; position=:topright)
    savefig(fig_cc, joinpath(figdir, "fig_results_dispersive_comparison_convergence.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_dispersive_comparison_convergence.pdf")

    # Print summary stats for text
    @printf("  Dispersive: J0=%.3e → J_final=%.3e (%.2f%% reduction)\n",
            loss_d[1], loss_d[end], 100.0*(1-loss_d[end]/loss_d[1]))
    @printf("  Non-disp (ε_s): J0=%.3e → J_final=%.3e (%.2f%% reduction)\n",
            loss_s[1], loss_s[end], 100.0*(1-loss_s[end]/loss_s[1]))
    @printf("  Non-disp (ε∞): J0=%.3e → J_final=%.3e (%.2f%% reduction)\n",
            loss_n[1], loss_n[end], 100.0*(1-loss_n[end]/loss_n[1]))
    @printf("  ε_s/dispersive initial ratio: %.1f×\n", loss_s[1]/loss_d[1])
    @printf("  ε∞/dispersive initial ratio: %.1f×\n", loss_n[1]/loss_d[1])
else
    println("  SKIPPED: missing data files")
    isfile(conv_disp) || println("    Missing: $conv_disp")
    isfile(conv_nondisp) || println("    Missing: $conv_nondisp")
    isfile(conv_eps_s) || println("    Missing: $conv_eps_s")
end

# ══════════════════════════════════════════════════════════════════════════
# M3: Two-anomaly model figures
# ══════════════════════════════════════════════════════════════════════════
println("\n=== M3: Two-anomaly model ===")

# Figure M3a: Convergence
conv_two = joinpath(datadir, "fwi_two_anomaly_convergence.csv")
if isfile(conv_two)
    raw_t = read_csv(conv_two)
    iter_t = Float64.(raw_t[:, 1])
    loss_t = Float64.(raw_t[:, 2])

    fig_tc = plot_scatter(iter_t, loss_t;
        xlabel="Iteration", ylabel="Regularized Objective",
        mode="lines+markers", color=COLORS[1], dash=DASHES[1],
        legend="Two-anomaly FWI", linewidth=2, marker_size=4,
        yscale="log")
    set_legend!(fig_tc; position=:topright)
    savefig(fig_tc, joinpath(figdir, "fig_results_two_anomaly_convergence.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_two_anomaly_convergence.pdf")
    @printf("  J0=%.3e → J_final=%.3e (%.2f%% reduction)\n",
            loss_t[1], loss_t[end], 100.0*(1-loss_t[end]/loss_t[1]))
else
    println("  SKIPPED: $conv_two not found")
end

# Figure M3b: 2D reconstruction (true, initial, estimated)
files_2d = [
    ("true", joinpath(datadir, "fwi_two_anomaly_reconstruction_2d_true.csv")),
    ("initial", joinpath(datadir, "fwi_two_anomaly_reconstruction_2d_initial.csv")),
    ("estimated", joinpath(datadir, "fwi_two_anomaly_reconstruction_2d_estimated.csv")),
]

if all(isfile(f[2]) for f in files_2d)
    for (tag, fpath) in files_2d
        x, y, U = read_2d_csv(fpath)
        fig_2d = plot_heatmap(x, y, U;
            xlabel="x [cm]", ylabel="Depth [cm]",
            colorscale="Viridis", zrange=[1.0, 16.0],
            yrange=[maximum(y), minimum(y)])
        savefig(fig_2d, joinpath(figdir, "fig_results_two_anomaly_2d_$(tag).pdf");
                width=336, height=IEEE_SINGLE_COL_H)
        println("  Saved fig_results_two_anomaly_2d_$(tag).pdf")
    end
else
    println("  SKIPPED: 2D data not found")
end

# Figure M3c: 1D reconstruction slices (pipe and void)
recon_pipe = joinpath(datadir, "fwi_two_anomaly_reconstruction_1d_pipe.csv")
recon_void = joinpath(datadir, "fwi_two_anomaly_reconstruction_1d_void.csv")

if isfile(recon_pipe) && isfile(recon_void)
    raw_p = read_csv(recon_pipe)
    raw_v = read_csv(recon_void)

    depth_p = Float64.(raw_p[:, 1])
    eps_true_p = Float64.(raw_p[:, 2])
    eps_init_p = Float64.(raw_p[:, 3])
    eps_est_p  = Float64.(raw_p[:, 4])

    depth_v = Float64.(raw_v[:, 1])
    eps_true_v = Float64.(raw_v[:, 2])
    eps_init_v = Float64.(raw_v[:, 3])
    eps_est_v  = Float64.(raw_v[:, 4])

    # Pipe slice
    fig_p = plot_scatter(depth_p, eps_true_p;
        xlabel="Depth [cm]", ylabel=raw"$\varepsilon_\infty$",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="True", linewidth=2)
    plot_scatter!(fig_p, depth_p, eps_init_p;
        color=COLORS[3], dash=DASHES[3], mode="lines",
        legend="Initial", linewidth=2)
    plot_scatter!(fig_p, depth_p, eps_est_p;
        color=COLORS[2], dash=DASHES[2], mode="lines+markers",
        legend="Estimated", linewidth=2, marker_size=4)
    set_legend!(fig_p; position=:topright)
    savefig(fig_p, joinpath(figdir, "fig_results_two_anomaly_1d_pipe.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_two_anomaly_1d_pipe.pdf")

    # Void slice
    fig_v = plot_scatter(depth_v, eps_true_v;
        xlabel="Depth [cm]", ylabel=raw"$\varepsilon_\infty$",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="True", linewidth=2)
    plot_scatter!(fig_v, depth_v, eps_init_v;
        color=COLORS[3], dash=DASHES[3], mode="lines",
        legend="Initial", linewidth=2)
    plot_scatter!(fig_v, depth_v, eps_est_v;
        color=COLORS[2], dash=DASHES[2], mode="lines+markers",
        legend="Estimated", linewidth=2, marker_size=4)
    set_legend!(fig_v; position=:topright)
    savefig(fig_v, joinpath(figdir, "fig_results_two_anomaly_1d_void.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_two_anomaly_1d_void.pdf")
else
    println("  SKIPPED: 1D data not found")
end

println("\n=== M1/M3 figure generation complete ===")
