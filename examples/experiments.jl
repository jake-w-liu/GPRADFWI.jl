using GPRADFWI
using Enzyme
using LinearAlgebra
using Printf

# This smoke experiment produces no manuscript figures or data files.
const PAPER_FIGURE_OUTPUTS = String[]

function main()
    config = create_config(nx=24, ny=20, dx=0.01, fc=300e6, npml=4,
                           src_ix=12, src_iy=10, rx_iy=10,
                           rx_ix_list=[8, 12, 16], nt=300)
    source = create_source(config)

    eps_background = fill(4.0, config.nx, config.ny)
    eps_true = copy(eps_background)
    eps_true[10:14, 8:12] .= 5.5
    deps = fill(1.5, config.nx, config.ny)
    tau = fill(0.4e-9, config.nx, config.ny)
    sigma = fill(1e-3, config.nx, config.ny)
    observed = run_forward!(config, eps_true, deps, tau, sigma, source)

    mask = falses(config.nx, config.ny)
    mask[10:14, 8:12] .= true
    params = fill(4.0, count(mask))
    objective = x -> forward_misfit(x, config, observed, source,
                                    eps_background, deps, tau, sigma,
                                    mask, :eps_inf)

    gradient = zeros(length(params))
    Enzyme.autodiff(
        Enzyme.Reverse,
        forward_misfit,
        Enzyme.Active,
        Enzyme.Duplicated(params, gradient),
        Enzyme.Const(config),
        Enzyme.Const(observed),
        Enzyme.Const(source),
        Enzyme.Const(eps_background),
        Enzyme.Const(deps),
        Enzyme.Const(tau),
        Enzyme.Const(sigma),
        Enzyme.Const(mask),
        Enzyme.Const(:eps_inf),
    )
    gradient_norm = norm(gradient)
    gradient_norm > 0.0 || error("smoke gradient is identically zero")
    direction = gradient ./ gradient_norm
    ad_directional = dot(gradient, direction)
    h = 3e-4
    fd_directional = (objective(params .+ h .* direction) -
                      objective(params .- h .* direction)) / (2h)
    relative_error = abs(ad_directional - fd_directional) /
                     max(abs(ad_directional), abs(fd_directional), eps(Float64))

    @printf("forward loss: %.6e\n", objective(params))
    @printf("AD/FD directional relative error: %.6e\n", relative_error)
    relative_error <= 1e-3 || error("AD/FD directional check failed")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
