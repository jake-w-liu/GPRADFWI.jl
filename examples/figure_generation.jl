# figure_generation.jl — Generate all publication-quality results figures
#
# Reads CSV data from paper/data/ and generates IEEE-quality figures
# using PlotlySupply.jl + PlotlyKaleido.jl (PDF export).
#
# Figures produced:
#   1. fig_results_bscan.pdf                 — GPR B-scan (selected receiver traces)
#   2. fig_results_material_profile.pdf      — Subsurface material model (1D)
#   3. fig_results_gradient_verification.pdf — AD vs FD gradient scatter
#   4. fig_results_fwi_convergence.pdf       — Multi-source FWI convergence
#   5. fig_results_fwi_reconstruction_1d.pdf — Large-domain reconstruction (1D)
#   6. fig_results_material_map_2d.pdf       — 2D eps_inf heatmap
#   7. fig_results_field_snapshot_*.pdf (x4) — Ez wavefield snapshots
#   8. fig_results_bscan_heatmap.pdf         — Full B-scan as heatmap
#   9. fig_results_fwi_reconstruction_2d_*.pdf (x3) — Large-domain 2D reconstruction
#  10. fig_results_gradient_sensitivity.pdf  — 2D gradient magnitude

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

# IEEE figure constants (colorblind-safe Wong palette)
const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
const DASHES = ["solid", "dash", "dashdot", "dot"]
const IEEE_SINGLE_COL_W = 504   # 3.5in at 144 DPI
const IEEE_SINGLE_COL_H = 360   # ~2.5in at 144 DPI
const IEEE_DOUBLE_COL_W = 1008  # 7.0in at 144 DPI

datadir = joinpath(@__DIR__, "..", "..", "paper", "data")
figdir  = joinpath(@__DIR__, "..", "..", "paper", "figs")
mkpath(figdir)

# ══════════════════════════════════════════════════════════════════════════
# Figure 1: GPR B-scan — Selected receiver traces
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_bscan.pdf ...")

bscan_file = joinpath(datadir, "validation_bscan.csv")
raw = readdlm(bscan_file, ','; comments=true, comment_char='#')
if raw[1, 1] isa AbstractString
    raw = raw[2:end, :]
end
time_ns = Float64.(raw[:, 1])
nrx = size(raw, 2) - 1

# Select 5 representative traces: near-offset, two mid-offsets, far-offset
trace_ids = round.(Int, range(1, nrx, length=5))

fig1 = plot_scatter(time_ns, Float64.(raw[:, trace_ids[1]+1]);
    xlabel="Time [ns]", ylabel="Ez [V/m]",
    mode="lines", color=COLORS[1], dash=DASHES[1],
    legend="Rx $(trace_ids[1])", linewidth=2)

for (k, tid) in enumerate(trace_ids[2:end])
    ci = mod(k, 4) + 1
    plot_scatter!(fig1, time_ns, Float64.(raw[:, tid+1]);
        color=COLORS[ci], dash=DASHES[ci], mode="lines",
        legend="Rx $tid", linewidth=2)
end

set_legend!(fig1; position=:topright)
savefig(fig1, joinpath(figdir, "fig_results_bscan.pdf");
        width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
println("  Saved fig_results_bscan.pdf")

# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Subsurface material profile
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_material_profile.pdf ...")

mat_file = joinpath(datadir, "validation_material_profile.csv")
raw_mat = readdlm(mat_file, ','; comments=true, comment_char='#')
if raw_mat[1, 1] isa AbstractString
    raw_mat = raw_mat[2:end, :]
end
depth_m  = Float64.(raw_mat[:, 1])
eps_inf  = Float64.(raw_mat[:, 2])
delta_eps = Float64.(raw_mat[:, 3])

# Only show subsurface region (depth >= 0)
mask = depth_m .>= -0.01
depth_sub = depth_m[mask]
eps_inf_sub = eps_inf[mask]
delta_eps_sub = delta_eps[mask]

fig2 = plot_scatter(depth_sub .* 100, eps_inf_sub;
    xlabel="Depth [cm]", ylabel="Relative Permittivity",
    mode="lines", color=COLORS[1], dash=DASHES[1],
    legend="eps_inf", linewidth=2)

plot_scatter!(fig2, depth_sub .* 100, delta_eps_sub;
    color=COLORS[2], dash=DASHES[2], mode="lines",
    legend="delta_eps", linewidth=2)

# Total static permittivity
eps_s_sub = eps_inf_sub .+ delta_eps_sub
plot_scatter!(fig2, depth_sub .* 100, eps_s_sub;
    color=COLORS[3], dash=DASHES[3], mode="lines",
    legend="eps_s = eps_inf + delta_eps", linewidth=2)

set_legend!(fig2; position=:topright)
savefig(fig2, joinpath(figdir, "fig_results_material_profile.pdf");
        width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
println("  Saved fig_results_material_profile.pdf")

# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Gradient verification — AD vs FD scatter plot
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_gradient_verification.pdf ...")

grad_file_disp = joinpath(datadir, "gradient_comparison_dispersive.csv")
grad_file_nondisp = joinpath(datadir, "gradient_comparison.csv")
grad_file = isfile(grad_file_disp) ? grad_file_disp : grad_file_nondisp
case_label = isfile(grad_file_disp) ? "Dispersive Debye case" : "Non-dispersive control"
raw_g = readdlm(grad_file, ','; comments=true, comment_char='#')
if raw_g[1, 1] isa AbstractString
    raw_g = raw_g[2:end, :]
end
grad_fd = Float64.(raw_g[:, 2])
grad_ad = Float64.(raw_g[:, 3])

fig3 = plot_scatter(grad_fd, grad_ad;
    xlabel="Finite-Difference Gradient",
    ylabel="AD Gradient (Enzyme)",
    mode="markers", color=COLORS[1],
    marker_size=5, marker_symbol="circle",
    legend="AD vs FD ($case_label)")

# y = x reference line
gmin = min(minimum(grad_fd), minimum(grad_ad))
gmax = max(maximum(grad_fd), maximum(grad_ad))
margin = 0.1 * (gmax - gmin)
plot_scatter!(fig3, [gmin - margin, gmax + margin], [gmin - margin, gmax + margin];
    color=COLORS[2], dash="dash", mode="lines",
    legend="y = x", linewidth=2)

set_legend!(fig3; position=:topleft)
savefig(fig3, joinpath(figdir, "fig_results_gradient_verification.pdf");
        width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
println("  Saved fig_results_gradient_verification.pdf")

# ══════════════════════════════════════════════════════════════════════════
# Figure 4: FWI convergence history (log scale) — large-domain multi-source
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_fwi_convergence.pdf ...")

conv_file_fwi = joinpath(datadir, "fwi_large_convergence.csv")
if isfile(conv_file_fwi)
    raw_c = readdlm(conv_file_fwi, ','; comments=true, comment_char='#')
    if raw_c[1, 1] isa AbstractString
        raw_c = raw_c[2:end, :]
    end
    iters     = Float64.(raw_c[:, 1])
    loss_vals = Float64.(raw_c[:, 2])

    fig4 = plot_scatter(iters, loss_vals;
        xlabel="Iteration", ylabel="Regularized Objective",
        mode="lines+markers", color=COLORS[1], dash=DASHES[1],
        legend="Loss", linewidth=2, marker_size=4,
        yscale="log")

    set_legend!(fig4; position=:topright)
    savefig(fig4, joinpath(figdir, "fig_results_fwi_convergence.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_fwi_convergence.pdf")
else
    println("  SKIPPED (data not found: $conv_file_fwi)")
end

# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Permittivity reconstruction vs ground truth — large-domain 1D slice
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_fwi_reconstruction_1d.pdf ...")

recon1d_file_fwi = joinpath(datadir, "fwi_large_reconstruction_1d.csv")
if isfile(recon1d_file_fwi)
    raw_r = readdlm(recon1d_file_fwi, ','; comments=true, comment_char='#')
    if raw_r[1, 1] isa AbstractString
        raw_r = raw_r[2:end, :]
    end
    depth_cm    = Float64.(raw_r[:, 1])
    eps_true    = Float64.(raw_r[:, 2])
    eps_init    = Float64.(raw_r[:, 3])
    eps_est     = Float64.(raw_r[:, 4])

    fig5 = plot_scatter(depth_cm, eps_true;
        xlabel="Depth [cm]", ylabel="Relative Permittivity (eps_inf)",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="True", linewidth=2)

    plot_scatter!(fig5, depth_cm, eps_est;
        color=COLORS[2], dash=DASHES[2], mode="lines+markers",
        legend="Reconstructed", linewidth=2, marker_size=5)

    plot_scatter!(fig5, depth_cm, eps_init;
        color=COLORS[3], dash=DASHES[3], mode="lines",
        legend="Initial", linewidth=2)

    set_legend!(fig5; position=:topright)
    savefig(fig5, joinpath(figdir, "fig_results_fwi_reconstruction_1d.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_fwi_reconstruction_1d.pdf")
else
    println("  SKIPPED (data not found: $recon1d_file_fwi)")
end

# ══════════════════════════════════════════════════════════════════════════
# Helper: Read 3-column (x, y, value) CSV into vectors + matrix
# ══════════════════════════════════════════════════════════════════════════

"""
Read a flat three-column CSV (x, y, value) and reshape to (x_vec, y_vec, U_matrix).
U is indexed as U[ix, jy] matching PlotlySupply's plot_heatmap convention.
"""
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
# Figure 6: 2D Material Map (eps_inf heatmap)
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_material_map_2d.pdf ...")

mat2d_file = joinpath(datadir, "material_map_2d.csv")
if isfile(mat2d_file)
    x_mat, y_mat, eps_2d = read_2d_csv(mat2d_file)
    # Convert depth to cm for display
    y_mat_cm = y_mat .* 100.0
    fig6 = plot_heatmap(x_mat .* 100.0, y_mat_cm, eps_2d;
        xlabel="x [cm]", ylabel="Depth [cm]",
        colorscale="Viridis",
        yrange=[maximum(y_mat_cm), minimum(y_mat_cm)])  # depth increases downward
    savefig(fig6, joinpath(figdir, "fig_results_material_map_2d.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_material_map_2d.pdf")
else
    println("  SKIPPED (data not found: $mat2d_file)")
end

# ══════════════════════════════════════════════════════════════════════════
# Figure 7: Ez Field Snapshots (4 separate PDFs for LaTeX subfigure)
# ══════════════════════════════════════════════════════════════════════════
println("Generating field snapshot figures ...")

snap_times = [2, 5, 8, 12]
snap_files = [@sprintf("field_snapshot_t%02dns.csv", t) for t in snap_times]

# First pass: find vmax across all snapshots for consistent color scale
snap_vmax = let vm = 0.0
    for fname in snap_files
        fpath = joinpath(datadir, fname)
        if isfile(fpath)
            _, _, Ez_snap = read_2d_csv(fpath)
            abs_vals = sort(abs.(Ez_snap[:]))
            # 99th percentile to avoid source singularity domination
            p99 = abs_vals[round(Int, 0.99 * length(abs_vals))]
            vm = max(vm, p99)
        end
    end
    vm
end

for (k, fname) in enumerate(snap_files)
    fpath = joinpath(datadir, fname)
    if isfile(fpath)
        x_s, y_s, Ez_snap = read_2d_csv(fpath)
        y_s_cm = y_s .* 100.0
        fig_snap = plot_heatmap(x_s .* 100.0, y_s_cm, Ez_snap;
            xlabel="x [cm]", ylabel="Depth [cm]",
            colorscale="RdBu",
            zrange=[-snap_vmax, snap_vmax],
            yrange=[maximum(y_s_cm), minimum(y_s_cm)],
            title="t = $(snap_times[k]) ns")
        outname = @sprintf("fig_results_field_snapshot_t%02dns.pdf", snap_times[k])
        savefig(fig_snap, joinpath(figdir, outname);
                width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
        println("  Saved $outname")
    else
        println("  SKIPPED (data not found: $fpath)")
    end
end

# ══════════════════════════════════════════════════════════════════════════
# Figure 8: Full B-scan Heatmap (all 91 receivers)
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_bscan_heatmap.pdf ...")

bscan_axes_file = joinpath(datadir, "bscan_heatmap_axes.csv")
if isfile(bscan_axes_file) && isfile(bscan_file)
    # Read receiver positions
    raw_ax = readdlm(bscan_axes_file, ','; comments=true, comment_char='#')
    if raw_ax[1, 1] isa AbstractString
        raw_ax = raw_ax[2:end, :]
    end
    rx_x_m = Float64.(raw_ax[:, 2])

    # B-scan data is already loaded above (time_ns, nrx from Figure 1)
    # Build matrix: (nrx × nt) for plot_heatmap(rx_positions, time, data)
    bscan_matrix = zeros(nrx, length(time_ns))
    for r in 1:nrx
        bscan_matrix[r, :] = Float64.(raw[:, r+1])
    end

    bmax = maximum(abs.(bscan_matrix))
    fig8 = plot_heatmap(rx_x_m .* 100.0, time_ns, bscan_matrix;
        xlabel="Receiver Position [cm]", ylabel="Time [ns]",
        colorscale="RdBu",
        zrange=[-bmax, bmax],
        yrange=[maximum(time_ns), 0.0])  # time increases downward
    savefig(fig8, joinpath(figdir, "fig_results_bscan_heatmap.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_bscan_heatmap.pdf")
else
    println("  SKIPPED (data not found)")
end

# ══════════════════════════════════════════════════════════════════════════
# Figure 9: 2D Reconstruction Comparison — large-domain multi-source FWI
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_fwi_reconstruction_2d.pdf ...")

recon_true_file = joinpath(datadir, "fwi_large_reconstruction_2d_true.csv")
recon_init_file = joinpath(datadir, "fwi_large_reconstruction_2d_initial.csv")
recon_est_file  = joinpath(datadir, "fwi_large_reconstruction_2d_estimated.csv")

if isfile(recon_true_file) && isfile(recon_init_file) && isfile(recon_est_file)
    x_t, y_t, eps_true_2d = read_2d_csv(recon_true_file)
    x_i, y_i, eps_init_2d = read_2d_csv(recon_init_file)
    x_e, y_e, eps_est_2d  = read_2d_csv(recon_est_file)

    # Shared color range: [background, pipe_peak] with margin
    zmin = min(minimum(eps_true_2d), minimum(eps_est_2d), minimum(eps_init_2d)) - 0.5
    zmax = max(maximum(eps_true_2d), maximum(eps_est_2d), maximum(eps_init_2d)) + 0.5

    # Generate 3 separate PDFs (composed via \subfloat in LaTeX)
    for (label, x_v, y_v, eps_v) in [("true", x_t, y_t, eps_true_2d),
                                       ("initial", x_i, y_i, eps_init_2d),
                                       ("estimated", x_e, y_e, eps_est_2d)]
        fig_r = plot_heatmap(x_v, y_v, eps_v;
            xlabel="x [cm]", ylabel="Depth [cm]",
            colorscale="Viridis", zrange=[zmin, zmax],
            yrange=[maximum(y_v), minimum(y_v)])
        outname = "fig_results_fwi_reconstruction_2d_$(label).pdf"
        savefig(fig_r, joinpath(figdir, outname);
                width=336, height=IEEE_SINGLE_COL_H)
        println("  Saved $outname")
    end
else
    println("  SKIPPED (reconstruction data not found)")
end

# ══════════════════════════════════════════════════════════════════════════
# Figure 10: Gradient Sensitivity Map (2D)
# ══════════════════════════════════════════════════════════════════════════
println("Generating fig_results_gradient_sensitivity.pdf ...")

grad_sens_file = joinpath(datadir, "gradient_sensitivity_2d.csv")
if isfile(grad_sens_file)
    x_g, y_g, grad_mag_2d = read_2d_csv(grad_sens_file)
    fig10 = plot_heatmap(x_g, y_g, grad_mag_2d;
        xlabel="x [cm]", ylabel="Depth [cm]",
        colorscale="Viridis",
        yrange=[maximum(y_g), minimum(y_g)])
    savefig(fig10, joinpath(figdir, "fig_results_gradient_sensitivity.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved fig_results_gradient_sensitivity.pdf")
else
    println("  SKIPPED (data not found: $grad_sens_file)")
end

println("\n=== All results figures generated ===")
