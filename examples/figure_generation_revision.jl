# figure_generation_revision.jl — Generate figures for revision experiments
#
# New figures for C1 (dispersive vs. non-dispersive) and M1 (noise robustness):
#   1. fig_results_dispersive_comparison_convergence.pdf  — Convergence overlay
#   2. fig_results_dispersive_comparison_1d.pdf           — 1D reconstruction overlay
#   3. fig_results_dispersive_comparison_2d.pdf           — 2D reconstruction (non-dispersive)
#   4. fig_results_noise_robustness_1d.pdf                — 1D reconstruction at 3 SNR levels
#   5. fig_results_noise_robustness_convergence.pdf       — Convergence at 3 SNR levels

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "PlotlySupply.jl"))

using PlotlySupply
using PlotlyKaleido: PlotlyKaleido
import PlotlyKaleido: savefig
using DelimitedFiles
using Printf

PlotlyKaleido.start(mathjax=false)

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
const DASHES = ["solid", "dash", "dashdot", "dot"]
const IEEE_SINGLE_COL_W = 504
const IEEE_SINGLE_COL_H = 360
const IEEE_DOUBLE_COL_W = 1008

datadir = joinpath(@__DIR__, "..", "..", "paper", "data")
figdir  = joinpath(@__DIR__, "..", "..", "paper", "figs")
mkpath(figdir)

# Helper to read CSV
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
# C1: Dispersive vs. Non-Dispersive Comparison
# ══════════════════════════════════════════════════════════════════════════

# Figure C1a: Convergence comparison
println("Generating fig_results_dispersive_comparison_convergence.pdf ...")
conv_disp = joinpath(datadir, "fwi_large_convergence.csv")
conv_nondisp = joinpath(datadir, "fwi_nondispersive_convergence.csv")

if isfile(conv_disp) && isfile(conv_nondisp)
    raw_d = read_csv(conv_disp)
    raw_n = read_csv(conv_nondisp)

    iter_d = Float64.(raw_d[:, 1])
    loss_d = Float64.(raw_d[:, 2])
    iter_n = Float64.(raw_n[:, 1])
    loss_n = Float64.(raw_n[:, 2])

    fig_cc = plot_scatter(iter_d, loss_d;
        xlabel="Iteration", ylabel="Regularized Objective",
        mode="lines+markers", color=COLORS[1], dash=DASHES[1],
        legend="Dispersive FWI", linewidth=2, marker_size=4,
        yscale="log")

    plot_scatter!(fig_cc, iter_n, loss_n;
        color=COLORS[2], dash=DASHES[2], mode="lines+markers",
        legend="Non-dispersive FWI", linewidth=2, marker_size=4)

    set_legend!(fig_cc; position=:topright)
    savefig(fig_cc, joinpath(figdir, "fig_results_dispersive_comparison_convergence.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_dispersive_comparison_convergence.pdf")
else
    println("  SKIPPED (data not found)")
end

# Figure C1b: 1D reconstruction comparison
println("Generating fig_results_dispersive_comparison_1d.pdf ...")
recon_disp = joinpath(datadir, "fwi_large_reconstruction_1d.csv")
recon_nondisp = joinpath(datadir, "fwi_nondispersive_reconstruction_1d.csv")

if isfile(recon_disp) && isfile(recon_nondisp)
    raw_d = read_csv(recon_disp)
    raw_n = read_csv(recon_nondisp)

    depth = Float64.(raw_d[:, 1])
    eps_true = Float64.(raw_d[:, 2])
    eps_disp = Float64.(raw_d[:, 4])
    eps_nondisp = Float64.(raw_n[:, 4])
    eps_init = Float64.(raw_d[:, 3])

    fig_cr = plot_scatter(depth, eps_true;
        xlabel="Depth [cm]", ylabel="Relative Permittivity (eps_inf)",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="True", linewidth=2)

    plot_scatter!(fig_cr, depth, eps_disp;
        color=COLORS[2], dash=DASHES[2], mode="lines+markers",
        legend="Dispersive FWI", linewidth=2, marker_size=4)

    plot_scatter!(fig_cr, depth, eps_nondisp;
        color=COLORS[4], dash=DASHES[4], mode="lines+markers",
        legend="Non-dispersive FWI", linewidth=2, marker_size=4,
        marker_symbol="diamond")

    plot_scatter!(fig_cr, depth, eps_init;
        color=COLORS[3], dash=DASHES[3], mode="lines",
        legend="Initial", linewidth=2)

    set_legend!(fig_cr; position=:topright)
    savefig(fig_cr, joinpath(figdir, "fig_results_dispersive_comparison_1d.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_dispersive_comparison_1d.pdf")
else
    println("  SKIPPED (data not found)")
end

# Figure C1c: 2D non-dispersive reconstruction (for side-by-side with existing)
println("Generating fig_results_reconstruction_2d_nondispersive.pdf ...")
recon_2d_nondisp = joinpath(datadir, "fwi_nondispersive_reconstruction_2d_estimated.csv")

if isfile(recon_2d_nondisp)
    # Read dispersive 2D for consistent color range
    recon_2d_true = joinpath(datadir, "fwi_large_reconstruction_2d_true.csv")
    recon_2d_disp = joinpath(datadir, "fwi_large_reconstruction_2d_estimated.csv")

    x_n, y_n, eps_n = read_2d_csv(recon_2d_nondisp)

    zmin = 3.5
    zmax = 16.0
    if isfile(recon_2d_true)
        _, _, eps_t = read_2d_csv(recon_2d_true)
        zmin = min(minimum(eps_t), minimum(eps_n)) - 0.5
        zmax = max(maximum(eps_t), maximum(eps_n)) + 0.5
    end

    fig_n2d = plot_heatmap(x_n, y_n, eps_n;
        xlabel="x [cm]", ylabel="Depth [cm]",
        colorscale="Viridis", zrange=[zmin, zmax],
        yrange=[maximum(y_n), minimum(y_n)])
    savefig(fig_n2d, joinpath(figdir, "fig_results_reconstruction_2d_nondispersive.pdf");
            width=336, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_reconstruction_2d_nondispersive.pdf")
else
    println("  SKIPPED (data not found)")
end

# ══════════════════════════════════════════════════════════════════════════
# M1: Noise Robustness
# ══════════════════════════════════════════════════════════════════════════

# Figure M1a: 1D reconstruction at 3 SNR levels overlaid with true and clean
println("Generating fig_results_noise_robustness_1d.pdf ...")

snr_levels = [40, 30, 20]
snr_labels = ["SNR = 40 dB", "SNR = 30 dB", "SNR = 20 dB"]
snr_colors = [COLORS[2], COLORS[3], COLORS[4]]
snr_dashes = [DASHES[2], DASHES[3], DASHES[4]]

recon_clean = joinpath(datadir, "fwi_large_reconstruction_1d.csv")
any_noisy = any(isfile(joinpath(datadir, "fwi_noisy_snr$(s)db_reconstruction_1d.csv")) for s in snr_levels)
multiseed_recon_files = [joinpath(datadir, "fwi_noisy_multiseed_reconstruction_stats_snr$(s)db.csv") for s in snr_levels]
has_multiseed_recon = all(isfile(f) for f in multiseed_recon_files)
multiseed_clean = joinpath(datadir, "fwi_noisy_multiseed_clean_convergence.csv")

if has_multiseed_recon
    # New multi-seed statistics (mean profile per SNR)
    raw_ref = read_csv(multiseed_recon_files[1])
    depth = Float64.(raw_ref[:, 1])
    eps_true = Float64.(raw_ref[:, 2])
    eps_init = Float64.(raw_ref[:, 3])
    eps_clean = Float64.(raw_ref[:, 4])

    fig_nr = plot_scatter(depth, eps_true;
        xlabel="Depth [cm]", ylabel="Relative Permittivity (eps_inf)",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="True", linewidth=2)

    plot_scatter!(fig_nr, depth, eps_clean;
        color="#555555", dash="solid", mode="lines",
        legend="Clean", linewidth=1)

    plot_scatter!(fig_nr, depth, eps_init;
        color=COLORS[3], dash=DASHES[3], mode="lines",
        legend="Initial", linewidth=1)

    for (k, fpath) in enumerate(multiseed_recon_files)
        local raw_n = read_csv(fpath)
        eps_noisy_mean = Float64.(raw_n[:, 5])
        plot_scatter!(fig_nr, depth, eps_noisy_mean;
            color=snr_colors[k], dash=snr_dashes[k], mode="lines+markers",
            legend=snr_labels[k], linewidth=2, marker_size=3)
    end

    set_legend!(fig_nr; position=:topright)
    savefig(fig_nr, joinpath(figdir, "fig_results_noise_robustness_1d.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_noise_robustness_1d.pdf (multi-seed means)")
elseif isfile(recon_clean) && any_noisy
    raw_c = read_csv(recon_clean)
    depth = Float64.(raw_c[:, 1])
    eps_true = Float64.(raw_c[:, 2])

    fig_nr = plot_scatter(depth, eps_true;
        xlabel="Depth [cm]", ylabel="Relative Permittivity (eps_inf)",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="True", linewidth=2)

    # Clean FWI (no noise, from existing results)
    eps_clean = Float64.(raw_c[:, 4])
    plot_scatter!(fig_nr, depth, eps_clean;
        color="#555555", dash="solid", mode="lines",
        legend="Clean", linewidth=1)

    for (k, snr) in enumerate(snr_levels)
        fpath = joinpath(datadir, "fwi_noisy_snr$(snr)db_reconstruction_1d.csv")
        if isfile(fpath)
            local raw_n = read_csv(fpath)
            eps_noisy = Float64.(raw_n[:, 4])
            plot_scatter!(fig_nr, depth, eps_noisy;
                color=snr_colors[k], dash=snr_dashes[k], mode="lines+markers",
                legend=snr_labels[k], linewidth=2, marker_size=3)
        end
    end

    set_legend!(fig_nr; position=:topright)
    savefig(fig_nr, joinpath(figdir, "fig_results_noise_robustness_1d.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_noise_robustness_1d.pdf")
else
    println("  SKIPPED (data not found)")
end

# Figure M1b: Convergence at 3 SNR levels
println("Generating fig_results_noise_robustness_convergence.pdf ...")

conv_clean = joinpath(datadir, "fwi_large_convergence.csv")
any_noisy_conv = any(isfile(joinpath(datadir, "fwi_noisy_snr$(s)db_convergence.csv")) for s in snr_levels)
multiseed_conv_stats = joinpath(datadir, "fwi_noisy_multiseed_convergence_stats.csv")
has_multiseed_conv = isfile(multiseed_conv_stats) && isfile(multiseed_clean)

if has_multiseed_conv
    raw_c = read_csv(multiseed_clean)
    iter_c = Float64.(raw_c[:, 1])
    loss_c = Float64.(raw_c[:, 2])

    fig_nc = plot_scatter(iter_c, loss_c;
        xlabel="Iteration", ylabel="Regularized Objective",
        mode="lines", color="#555555", dash="solid",
        legend="Clean", linewidth=1,
        yscale="log")

    raw_stats = read_csv(multiseed_conv_stats)
    snr_col = Int.(raw_stats[:, 1])
    iter_col = Float64.(raw_stats[:, 2])
    mean_col = Float64.(raw_stats[:, 3])
    for (k, snr) in enumerate(snr_levels)
        idx = findall(==(snr), snr_col)
        if !isempty(idx)
            plot_scatter!(fig_nc, iter_col[idx], mean_col[idx];
                color=snr_colors[k], dash=snr_dashes[k], mode="lines+markers",
                legend=snr_labels[k], linewidth=2, marker_size=3)
        end
    end

    set_legend!(fig_nc; position=:topright)
    savefig(fig_nc, joinpath(figdir, "fig_results_noise_robustness_convergence.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_noise_robustness_convergence.pdf (multi-seed means)")
elseif isfile(conv_clean) && any_noisy_conv
    raw_c = read_csv(conv_clean)
    iter_c = Float64.(raw_c[:, 1])
    loss_c = Float64.(raw_c[:, 2])

    fig_nc = plot_scatter(iter_c, loss_c;
        xlabel="Iteration", ylabel="Regularized Objective",
        mode="lines", color="#555555", dash="solid",
        legend="Clean", linewidth=1,
        yscale="log")

    for (k, snr) in enumerate(snr_levels)
        fpath = joinpath(datadir, "fwi_noisy_snr$(snr)db_convergence.csv")
        if isfile(fpath)
            local raw_n = read_csv(fpath)
            local iter_n = Float64.(raw_n[:, 1])
            local loss_n = Float64.(raw_n[:, 2])
            plot_scatter!(fig_nc, iter_n, loss_n;
                color=snr_colors[k], dash=snr_dashes[k], mode="lines+markers",
                legend=snr_labels[k], linewidth=2, marker_size=3)
        end
    end

    set_legend!(fig_nc; position=:topright)
    savefig(fig_nc, joinpath(figdir, "fig_results_noise_robustness_convergence.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_noise_robustness_convergence.pdf")
else
    println("  SKIPPED (data not found)")
end

println("\n=== Revision figures generated ===")
