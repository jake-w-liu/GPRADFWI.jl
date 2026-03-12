# revision_reduced_common.jl
# Shared reduced-domain setup for revision experiments (uncertainty + multi-seed noise).

using GPRADFWI
using Printf
using Statistics

function build_reduced_multisource_context()
    domain_x = 0.80
    domain_y = 0.64
    grid_dx = 0.01
    fc_gpr = 250e6

    nx = round(Int, domain_x / grid_dx)
    ny = round(Int, domain_y / grid_dx)
    npml = 8

    rx_y = npml + 6
    rx_x_list = collect((npml+2):3:(nx-npml-2))

    src_x_list = [18, 40, 62]
    nsrc = length(src_x_list)

    configs = FDTDConfig[]
    src_waveforms = Vector{Float64}[]
    for sx in src_x_list
        cfg = create_config(
            nx=nx, ny=ny, dx=grid_dx, fc=fc_gpr, npml=npml,
            src_ix=sx, src_iy=rx_y,
            rx_iy=rx_y, rx_ix_list=rx_x_list,
            nt=240,
        )
        push!(configs, cfg)
        push!(src_waveforms, create_source(cfg))
    end

    eps_inf_true = ones(nx, ny)
    deps_true = zeros(nx, ny)
    tau_true = zeros(nx, ny)
    sigma_true = zeros(nx, ny)

    surface_j = npml + 8
    layer1_top = surface_j
    layer1_bot = surface_j + 22  # ~22 cm thick

    for j in layer1_top:min(layer1_bot, ny), i in 1:nx
        eps_inf_true[i, j] = 4.0
        deps_true[i, j] = 4.0
        tau_true[i, j] = 0.3e-9
        sigma_true[i, j] = 0.005
    end

    layer2_top = layer1_bot + 1
    for j in layer2_top:ny, i in 1:nx
        eps_inf_true[i, j] = 6.0
        deps_true[i, j] = 10.0
        tau_true[i, j] = 1.0e-9
        sigma_true[i, j] = 0.02
    end

    pipe_cx = nx ÷ 2
    pipe_cy = surface_j + 14
    pipe_r = 4

    for j in 1:ny, i in 1:nx
        if (i - pipe_cx)^2 + (j - pipe_cy)^2 <= pipe_r^2
            eps_inf_true[i, j] = 14.0
            deps_true[i, j] = 8.0
            tau_true[i, j] = 0.5e-9
            sigma_true[i, j] = 0.001
        end
    end

    inv_x_lo = pipe_cx - 10
    inv_x_hi = pipe_cx + 10
    inv_y_lo = pipe_cy - 8
    inv_y_hi = pipe_cy + 8

    param_mask = falses(nx, ny)
    for j in inv_y_lo:inv_y_hi, i in inv_x_lo:inv_x_hi
        param_mask[i, j] = true
    end

    obs_datas_clean = Matrix{Float64}[]
    for k in 1:nsrc
        od = run_forward!(configs[k], eps_inf_true, deps_true, tau_true, sigma_true, src_waveforms[k])
        push!(obs_datas_clean, od)
    end

    return (
        nx=nx,
        ny=ny,
        grid_dx=grid_dx,
        npml=npml,
        rx_y=rx_y,
        rx_x_list=rx_x_list,
        src_x_list=src_x_list,
        configs=configs,
        src_waveforms=src_waveforms,
        eps_inf_true=eps_inf_true,
        deps_true=deps_true,
        tau_true=tau_true,
        sigma_true=sigma_true,
        surface_j=surface_j,
        layer1_top=layer1_top,
        layer1_bot=layer1_bot,
        layer2_top=layer2_top,
        pipe_cx=pipe_cx,
        pipe_cy=pipe_cy,
        pipe_r=pipe_r,
        inv_x_lo=inv_x_lo,
        inv_x_hi=inv_x_hi,
        inv_y_lo=inv_y_lo,
        inv_y_hi=inv_y_hi,
        param_mask=param_mask,
        n_params=count(param_mask),
        obs_datas_clean=obs_datas_clean,
    )
end

function build_initial_model(ctx; layer_shift_cells::Int=0)
    nx, ny = ctx.nx, ctx.ny

    eps_inf = ones(nx, ny)
    deps = zeros(nx, ny)
    tau = zeros(nx, ny)
    sigma = zeros(nx, ny)

    layer1_top = ctx.layer1_top
    layer1_bot = min(ctx.layer1_bot + layer_shift_cells, ny-1)
    layer2_top = layer1_bot + 1

    for j in layer1_top:layer1_bot, i in 1:nx
        eps_inf[i, j] = 4.0
        deps[i, j] = 4.0
        tau[i, j] = 0.3e-9
        sigma[i, j] = 0.005
    end

    for j in layer2_top:ny, i in 1:nx
        eps_inf[i, j] = 6.0
        deps[i, j] = 10.0
        tau[i, j] = 1.0e-9
        sigma[i, j] = 0.02
    end

    return eps_inf, deps, tau, sigma
end

function compute_region_metrics(ctx, eps_est::Matrix{Float64})
    eps_true_region = Float64[]
    eps_est_region = Float64[]
    for j in ctx.inv_y_lo:ctx.inv_y_hi, i in ctx.inv_x_lo:ctx.inv_x_hi
        push!(eps_true_region, ctx.eps_inf_true[i, j])
        push!(eps_est_region, eps_est[i, j])
    end

    rmse = sqrt(mean((eps_true_region .- eps_est_region).^2))
    peak_true = maximum(ctx.eps_inf_true[ctx.inv_x_lo:ctx.inv_x_hi, ctx.inv_y_lo:ctx.inv_y_hi])
    peak_est = maximum(eps_est[ctx.inv_x_lo:ctx.inv_x_hi, ctx.inv_y_lo:ctx.inv_y_hi])
    recovery = 100.0 * peak_est / peak_true

    return rmse, peak_true, peak_est, recovery
end

function save_convergence_csv(path::String, result; header::String="")
    open(path, "w") do io
        if !isempty(header)
            write(io, "# $header\n")
        end
        write(io, "iteration,loss,grad_norm\n")
        for k in eachindex(result.loss_history)
            @printf(io, "%d,%.12e,%.12e\n",
                    k - 1,
                    result.loss_history[k],
                    result.grad_norm_history[k])
        end
    end
end

function save_reconstruction_slice_csv(path::String, ctx, eps_est::Matrix{Float64}, eps_init::Matrix{Float64}; header::String="")
    open(path, "w") do io
        if !isempty(header)
            write(io, "# $header\n")
        end
        write(io, "depth_cm,eps_inf_true,eps_inf_initial,eps_inf_estimated\n")
        for j in ctx.inv_y_lo:ctx.inv_y_hi
            depth_cm = (j - ctx.surface_j) * ctx.grid_dx * 100.0
            @printf(io, "%.2f,%.6f,%.6f,%.6f\n",
                    depth_cm,
                    ctx.eps_inf_true[ctx.pipe_cx, j],
                    eps_init[ctx.pipe_cx, j],
                    eps_est[ctx.pipe_cx, j])
        end
    end
end
