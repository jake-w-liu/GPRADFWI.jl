# inversion.jl — FWI gradient computation (AD + finite-difference) and optimization
#
# Two gradient methods:
#   1. Finite-difference (reference, always works)
#   2. Enzyme.jl reverse-mode AD (the novel contribution)

using Enzyme

"""
    fd_gradient(f, x; h=1e-5)

Compute gradient of scalar function `f(x)` via central finite differences.
Reference implementation for validating AD gradients.

Uses step size h with relative scaling: h_i = h * max(|x_i|, 1).
"""
function fd_gradient(f::Function, x::Vector{Float64}; h::Float64=1e-5)
    n = length(x)
    grad = Vector{Float64}(undef, n)
    x_p = copy(x)
    x_m = copy(x)
    for i in 1:n
        hi = h * max(abs(x[i]), 1.0)
        x_p[i] = x[i] + hi
        x_m[i] = x[i] - hi
        grad[i] = (f(x_p) - f(x_m)) / (2.0 * hi)
        x_p[i] = x[i]
        x_m[i] = x[i]
    end
    return grad
end

"""
    ad_gradient(f, x)

Compute gradient of scalar function `f(x)` via Enzyme.jl reverse-mode AD.
"""
function ad_gradient(f::Function, x::Vector{Float64})
    dx = zeros(length(x))
    Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Duplicated(x, dx))
    return dx
end

"""
    _build_idx_map(param_mask)

Build a mapping from 2D grid coordinates to flat parameter vector indices.
Returns a matrix of the same size as `param_mask`, where entry (i,j) gives
the 1-based index into the flat vector `x`, or 0 if (i,j) is not inverted.
"""
function _build_idx_map(param_mask::BitMatrix)
    nx, ny = size(param_mask)
    idx_map = zeros(Int, nx, ny)
    k = 0
    for j in 1:ny, i in 1:nx
        if param_mask[i, j]
            k += 1
            idx_map[i, j] = k
        end
    end
    return idx_map
end

"""
    tikhonov_penalty(x, idx_map, param_mask; stride=1, param_idx=1)

Compute the Tikhonov (first-order smoothness) penalty:
    R(m) = Σ_{adjacent (i,j),(i',j')} (m_i - m_j)²
using forward differences in both grid directions.

For multi-parameter cells (stride > 1), `param_idx` selects which parameter
within each cell to penalize (1 = first parameter, 2 = second, etc.).
The flat vector is assumed to have `stride` entries per cell, with the
`param_idx`-th entry at position `(cell_index - 1) * stride + param_idx`.
"""
function tikhonov_penalty(x::Vector{Float64}, idx_map::Matrix{Int},
                          param_mask::BitMatrix;
                          stride::Int=1, param_idx::Int=1)
    nx, ny = size(param_mask)
    penalty = 0.0
    for j in 1:ny, i in 1:nx
        if !param_mask[i, j]
            continue
        end
        flat_i = (idx_map[i, j] - 1) * stride + param_idx
        xi = x[flat_i]
        # Forward difference in i-direction
        if i < nx && param_mask[i+1, j]
            flat_ip = (idx_map[i+1, j] - 1) * stride + param_idx
            penalty += (x[flat_ip] - xi)^2
        end
        # Forward difference in j-direction
        if j < ny && param_mask[i, j+1]
            flat_jp = (idx_map[i, j+1] - 1) * stride + param_idx
            penalty += (x[flat_jp] - xi)^2
        end
    end
    return penalty
end

"""
    tikhonov_gradient(x, idx_map, param_mask; stride=1, param_idx=1)

Compute the gradient of the Tikhonov penalty with respect to `x`.
This is the discrete negative Laplacian (graph Laplacian) applied to the
parameter field.

For multi-parameter cells (stride > 1), see `tikhonov_penalty` for layout.
"""
function tikhonov_gradient(x::Vector{Float64}, idx_map::Matrix{Int},
                           param_mask::BitMatrix;
                           stride::Int=1, param_idx::Int=1)
    nx, ny = size(param_mask)
    grad = zeros(length(x))
    for j in 1:ny, i in 1:nx
        if !param_mask[i, j]
            continue
        end
        flat_i = (idx_map[i, j] - 1) * stride + param_idx
        xi = x[flat_i]
        # Forward difference in i-direction
        if i < nx && param_mask[i+1, j]
            flat_ip = (idx_map[i+1, j] - 1) * stride + param_idx
            diff = x[flat_ip] - xi
            grad[flat_i]  -= 2.0 * diff
            grad[flat_ip] += 2.0 * diff
        end
        # Forward difference in j-direction
        if j < ny && param_mask[i, j+1]
            flat_jp = (idx_map[i, j+1] - 1) * stride + param_idx
            diff = x[flat_jp] - xi
            grad[flat_i]  -= 2.0 * diff
            grad[flat_jp] += 2.0 * diff
        end
    end
    return grad
end

"""
    run_fwi(config, obs_data, src_waveform, eps_inf_init, deps_map, tau_map,
            sigma_init, param_mask; max_iter, param_type, use_ad, verbose)

Run a minimal single-source full-waveform inversion using L-BFGS.

This helper optimizes only the data misfit returned by `forward_misfit`.
It does not add Tikhonov regularization or projected parameter bounds.
The manuscript-level inversion workflow and archived paper figures use
`run_fwi_multisource` via the example drivers under `GPRADFWI.jl/examples/`.

# Arguments
- `config`: FDTD configuration
- `obs_data`: observed receiver data (nt × nrx)
- `src_waveform`: source waveform
- `eps_inf_init, sigma_init`: initial model guesses
- `deps_map, tau_map`: fixed Debye parameters (or zeros for non-dispersive)
- `param_mask`: BitMatrix of cells to invert
- `max_iter`: maximum L-BFGS iterations (default 30)
- `param_type`: :eps_inf, :sigma, or :both
- `use_ad`: true for Enzyme AD, false for finite differences
- `verbose`: print progress
"""
function run_fwi(config::FDTDConfig,
                 obs_data::Matrix{Float64},
                 src_waveform::Vector{Float64},
                 eps_inf_init::Matrix{Float64},
                 deps_map::Matrix{Float64},
                 tau_map::Matrix{Float64},
                 sigma_init::Matrix{Float64},
                 param_mask::BitMatrix;
                 max_iter::Int = 30,
                 param_type::Symbol = :eps_inf,
                 use_ad::Bool = false,
                 verbose::Bool = true)
    nx = config.nx
    ny = config.ny

    # Pack initial parameters into flat vector
    x0 = Float64[]
    for j in 1:ny, i in 1:nx
        if param_mask[i, j]
            if param_type == :eps_inf || param_type == :both
                push!(x0, eps_inf_init[i, j])
            end
            if param_type == :sigma || param_type == :both
                push!(x0, sigma_init[i, j])
            end
        end
    end

    if verbose
        @printf("FWI: %d parameters, max %d iterations, param_type=%s\n",
                length(x0), max_iter, param_type)
        println("  Note: run_fwi is the minimal single-source helper; paper results use run_fwi_multisource with regularization and bounds.")
    end

    # Track convergence and per-iteration diagnostics
    loss_history = Float64[]
    grad_norm_history = Float64[]
    loss_data_history = Float64[]
    loss_reg_eps_history = Float64[]
    loss_reg_sigma_history = Float64[]
    step_alpha_history = Float64[]
    line_search_backtracks = Int[]

    # Objective function
    function objective(x)
        return forward_misfit(x, config, obs_data, src_waveform,
                              eps_inf_init, deps_map, tau_map, sigma_init,
                              param_mask, param_type)
    end

    # L-BFGS optimization with safeguarded line search
    x = copy(x0)
    step_size = 1.0

    # L-BFGS memory
    m = 5  # memory depth
    s_hist = Vector{Vector{Float64}}()  # s_k = x_{k+1} - x_k
    y_hist = Vector{Vector{Float64}}()  # y_k = g_{k+1} - g_k

    # Initial gradient
    grad = use_ad ? ad_gradient(objective, x) : fd_gradient(objective, x)
    f_val = objective(x)

    # Track best iterate
    x_best = copy(x)
    f_best = f_val

    push!(loss_history, f_val)
    push!(grad_norm_history, norm(grad))
    push!(loss_data_history, f_val)      # run_fwi has no explicit regularization term
    push!(loss_reg_eps_history, 0.0)
    push!(loss_reg_sigma_history, 0.0)
    push!(step_alpha_history, NaN)       # iteration 0 has no line-search step
    push!(line_search_backtracks, 0)

    if verbose
        @printf("  iter %3d: loss = %.6e, |grad| = %.6e\n", 0, f_val, norm(grad))
        flush(stdout)
    end

    for iter in 1:max_iter
        # L-BFGS two-loop recursion to compute search direction
        q = copy(grad)
        alphas = Float64[]
        k = length(s_hist)

        for ii in k:-1:1
            rho = 1.0 / dot(y_hist[ii], s_hist[ii])
            alpha_i = rho * dot(s_hist[ii], q)
            push!(alphas, alpha_i)
            q .-= alpha_i .* y_hist[ii]
        end
        reverse!(alphas)

        # Initial Hessian scaling
        if k > 0
            gamma = dot(s_hist[end], y_hist[end]) / dot(y_hist[end], y_hist[end])
            r = gamma .* q
        else
            r = q  # steepest descent for first iteration
        end

        for ii in 1:k
            rho = 1.0 / dot(y_hist[ii], s_hist[ii])
            beta = rho * dot(y_hist[ii], r)
            r .+= (alphas[ii] - beta) .* s_hist[ii]
        end

        direction = -r  # search direction

        # Verify descent direction; fall back to steepest descent if not
        dg = dot(grad, direction)
        if dg >= 0.0
            direction = -grad
            dg = -dot(grad, grad)
            empty!(s_hist)
            empty!(y_hist)
            if verbose
                @printf("  [L-BFGS reset: non-descent direction]\n")
                flush(stdout)
            end
        end

        # Backtracking line search (Armijo condition)
        alpha = step_size
        c1_armijo = 1e-4
        x_trial = x .+ alpha .* direction
        f_new = objective(x_trial)
        n_backtracks = 0

        ls_success = false
        for _ in 1:20
            if f_new <= f_val + c1_armijo * alpha * dg
                ls_success = true
                break
            end
            alpha *= 0.5
            n_backtracks += 1
            x_trial = x .+ alpha .* direction
            f_new = objective(x_trial)
        end

        # If line search failed, reject step and reset L-BFGS memory
        if !ls_success && f_new > f_val
            if verbose
                @printf("  [Line search failed, resetting L-BFGS]\n")
                flush(stdout)
            end
            empty!(s_hist)
            empty!(y_hist)
            gnorm = norm(grad)
            alpha = min(1e-2, 1.0 / gnorm)
            direction = -grad
            x_trial = x .+ alpha .* direction
            f_new = objective(x_trial)
        end

        # Update
        x_new = x_trial
        grad_new = use_ad ? ad_gradient(objective, x_new) : fd_gradient(objective, x_new)

        # Store L-BFGS pairs
        s_k = x_new .- x
        y_k = grad_new .- grad
        if dot(s_k, y_k) > 1e-20  # curvature condition
            push!(s_hist, s_k)
            push!(y_hist, y_k)
            if length(s_hist) > m
                popfirst!(s_hist)
                popfirst!(y_hist)
            end
        end

        x = x_new
        grad = grad_new
        f_val = f_new

        # Update best iterate
        if f_val < f_best
            x_best = copy(x)
            f_best = f_val
        end

        push!(loss_history, f_val)
        push!(grad_norm_history, norm(grad))
        push!(loss_data_history, f_val)
        push!(loss_reg_eps_history, 0.0)
        push!(loss_reg_sigma_history, 0.0)
        push!(step_alpha_history, alpha)
        push!(line_search_backtracks, n_backtracks)

        if verbose
            @printf("  iter %3d: loss = %.6e, |grad| = %.6e, α = %.2e\n",
                    iter, f_val, norm(grad), alpha)
            flush(stdout)
        end

        # Convergence check
        if norm(grad) < 1e-10 * max(abs(f_val), 1.0)
            verbose && println("  Converged (gradient norm).")
            break
        end
    end

    if verbose && f_best < f_val
        @printf("  Using best iterate (loss = %.6e vs final %.6e)\n", f_best, f_val)
        flush(stdout)
    end

    # Unpack best parameters (not necessarily the final iterate)
    x_final = f_best < f_val ? x_best : x
    eps_inf_est = copy(eps_inf_init)
    sigma_est = copy(sigma_init)
    idx = 1
    for j in 1:ny, i in 1:nx
        if param_mask[i, j]
            if param_type == :eps_inf || param_type == :both
                eps_inf_est[i, j] = x_final[idx]
                idx += 1
            end
            if param_type == :sigma || param_type == :both
                sigma_est[i, j] = x_final[idx]
                idx += 1
            end
        end
    end

    return FWIResult(eps_inf_est, deps_map, sigma_est, loss_history,
                     grad_norm_history, loss_data_history,
                     loss_reg_eps_history, loss_reg_sigma_history,
                     step_alpha_history, line_search_backtracks,
                     length(loss_history) - 1)
end

"""
    run_fwi_multisource(configs, obs_datas, src_waveforms, eps_inf_init, deps_map,
                         tau_map, sigma_init, param_mask; kwargs...)

Run multi-source full-waveform inversion using L-BFGS optimization.
Each source position has its own FDTDConfig and observed data. The total misfit
is the sum over all sources, and gradients are accumulated per-source to keep
peak memory at single-source level.

This is the bounded/regularized driver used by the paper example scripts,
including `run_fwi_large_domain.jl`, `run_fwi_joint.jl`,
`run_fwi_noisy_multiseed.jl`, and `run_fwi_uncertainty.jl`.

# Arguments
- `configs`: vector of FDTDConfig (one per source position)
- `obs_datas`: vector of observed data matrices (nt × nrx each)
- `src_waveforms`: vector of source waveforms
- `eps_inf_init, sigma_init`: initial model (shared across all sources)
- `deps_map, tau_map`: fixed Debye parameters
- `param_mask`: BitMatrix of cells to invert
- `max_iter`: maximum L-BFGS iterations (default 50)
- `param_type`: :eps_inf, :sigma, or :both
- `use_ad`: true for Enzyme AD, false for finite differences
- `verbose`: print progress
"""
function run_fwi_multisource(configs::Vector{FDTDConfig},
                              obs_datas::Vector{Matrix{Float64}},
                              src_waveforms::Vector{Vector{Float64}},
                              eps_inf_init::Matrix{Float64},
                              deps_map::Matrix{Float64},
                              tau_map::Matrix{Float64},
                              sigma_init::Matrix{Float64},
                              param_mask::BitMatrix;
                              max_iter::Int = 50,
                              param_type::Symbol = :eps_inf,
                              use_ad::Bool = true,
                              verbose::Bool = true,
                              lower_bound::Float64 = 1.0,
                              upper_bound::Float64 = 25.0,
                              lower_bound_sigma::Float64 = 0.0,
                              upper_bound_sigma::Float64 = 0.1,
                              lambda::Float64 = 0.0,
                              lambda_sigma::Float64 = -1.0,
                              lambda_damp::Float64 = 0.0,
                              lambda_damp_sigma::Float64 = -1.0)
    nsrc = length(configs)
    @assert length(obs_datas) == nsrc "Need one obs_data per source"
    @assert length(src_waveforms) == nsrc "Need one src_waveform per source"

    nx = configs[1].nx
    ny = configs[1].ny

    # Pack initial parameters into flat vector
    x0 = Float64[]
    for j in 1:ny, i in 1:nx
        if param_mask[i, j]
            if param_type == :eps_inf || param_type == :both
                push!(x0, eps_inf_init[i, j])
            end
            if param_type == :sigma || param_type == :both
                push!(x0, sigma_init[i, j])
            end
        end
    end

    # Build index map for Tikhonov regularization
    idx_map = _build_idx_map(param_mask)

    # Multi-parameter stride and bounds
    param_stride = (param_type == :both) ? 2 : 1
    n_cells = count(param_mask)
    lam_sig = lambda_sigma < 0 ? lambda : lambda_sigma  # default to lambda
    lam_d = lambda_damp
    lam_d_sig = lambda_damp_sigma < 0 ? lambda_damp : lambda_damp_sigma

    if param_type == :both
        lb_vec = Vector{Float64}(undef, 2 * n_cells)
        ub_vec = Vector{Float64}(undef, 2 * n_cells)
        for k in 1:n_cells
            lb_vec[2*(k-1)+1] = lower_bound
            ub_vec[2*(k-1)+1] = upper_bound
            lb_vec[2*(k-1)+2] = lower_bound_sigma
            ub_vec[2*(k-1)+2] = upper_bound_sigma
        end
    else
        lb_vec = fill(param_type == :sigma ? lower_bound_sigma : lower_bound, n_cells)
        ub_vec = fill(param_type == :sigma ? upper_bound_sigma : upper_bound, n_cells)
    end

    if verbose
        @printf("Multi-source FWI: %d sources, %d parameters, max %d iterations\n",
                nsrc, length(x0), max_iter)
        if lambda > 0
            @printf("  Tikhonov regularization: lambda = %.2e\n", lambda)
        end
        if param_type == :both && lam_sig != lambda
            @printf("  Tikhonov regularization (sigma): lambda_sigma = %.2e\n", lam_sig)
        end
        if lam_d > 0
            @printf("  Norm damping (eps_inf): lambda_damp = %.2e\n", lam_d)
        end
        if lam_d_sig > 0
            @printf("  Norm damping (sigma): lambda_damp_sigma = %.2e\n", lam_d_sig)
        end
        flush(stdout)
    end

    # Track convergence and per-iteration diagnostics
    loss_history = Float64[]
    grad_norm_history = Float64[]
    loss_data_history = Float64[]
    loss_reg_eps_history = Float64[]
    loss_reg_sigma_history = Float64[]
    step_alpha_history = Float64[]
    line_search_backtracks = Int[]

    # Multi-source objective diagnostics:
    #   total = data misfit + regularization (including optional norm damping)
    function objective_terms(x)
        loss_data = 0.0
        for k in 1:nsrc
            loss_data += forward_misfit(x, configs[k], obs_datas[k], src_waveforms[k],
                                        eps_inf_init, deps_map, tau_map, sigma_init,
                                        param_mask, param_type)
        end

        loss_reg_eps = 0.0
        loss_reg_sigma = 0.0

        # Tikhonov regularization (per-parameter for joint inversion)
        if lambda > 0
            loss_reg_eps += lambda * tikhonov_penalty(x, idx_map, param_mask;
                                                      stride=param_stride, param_idx=1)
        end
        if param_stride > 1 && lam_sig > 0
            loss_reg_sigma += lam_sig * tikhonov_penalty(x, idx_map, param_mask;
                                                         stride=param_stride, param_idx=2)
        end
        # Norm damping: penalize deviations from initial model
        if lam_d > 0 || lam_d_sig > 0
            for k in 1:length(x)
                if param_type == :both
                    if k % 2 == 1
                        loss_reg_eps += lam_d * (x[k] - x0[k])^2
                    else
                        loss_reg_sigma += lam_d_sig * (x[k] - x0[k])^2
                    end
                elseif param_type == :sigma
                    loss_reg_sigma += lam_d * (x[k] - x0[k])^2
                else
                    loss_reg_eps += lam_d * (x[k] - x0[k])^2
                end
            end
        end

        total = loss_data + loss_reg_eps + loss_reg_sigma
        return (total=total,
                data=loss_data,
                reg_eps=loss_reg_eps,
                reg_sigma=loss_reg_sigma)
    end

    function objective(x)
        return objective_terms(x).total
    end

    # Multi-source gradient: sum of per-source gradients
    # Uses direct Enzyme.autodiff on forward_misfit (not closures) so Enzyme
    # compiles the AD rules once and reuses them across all sources.
    function multi_gradient(x)
        g = zeros(length(x))
        for k in 1:nsrc
            if use_ad
                dx_k = zeros(length(x))
                Enzyme.autodiff(Enzyme.Reverse, forward_misfit, Enzyme.Active,
                    Enzyme.Duplicated(x, dx_k),
                    Enzyme.Const(configs[k]),
                    Enzyme.Const(obs_datas[k]),
                    Enzyme.Const(src_waveforms[k]),
                    Enzyme.Const(eps_inf_init),
                    Enzyme.Const(deps_map),
                    Enzyme.Const(tau_map),
                    Enzyme.Const(sigma_init),
                    Enzyme.Const(param_mask),
                    Enzyme.Const(param_type))
                g .+= dx_k
            else
                fk = xk -> forward_misfit(xk, configs[k], obs_datas[k], src_waveforms[k],
                                           eps_inf_init, deps_map, tau_map, sigma_init,
                                           param_mask, param_type)
                g .+= fd_gradient(fk, x)
            end
        end
        # Add Tikhonov regularization gradient (per-parameter for joint inversion)
        if lambda > 0
            g .+= lambda .* tikhonov_gradient(x, idx_map, param_mask;
                                               stride=param_stride, param_idx=1)
        end
        if param_stride > 1 && lam_sig > 0
            g .+= lam_sig .* tikhonov_gradient(x, idx_map, param_mask;
                                                stride=param_stride, param_idx=2)
        end
        # Norm damping gradient: 2 * lambda_damp * (x - x0)
        if lam_d > 0 || lam_d_sig > 0
            for k in 1:length(x)
                if param_type == :both
                    ld = (k % 2 == 1) ? lam_d : lam_d_sig
                else
                    ld = lam_d
                end
                g[k] += 2.0 * ld * (x[k] - x0[k])
            end
        end
        return g
    end

    # L-BFGS optimization
    x = copy(x0)
    step_size = 1.0
    m = 5  # memory depth
    s_hist = Vector{Vector{Float64}}()
    y_hist = Vector{Vector{Float64}}()

    # Initial gradient and objective
    grad = multi_gradient(x)
    terms = objective_terms(x)
    f_val = terms.total

    # Track best iterate (FWI can have non-monotone convergence)
    x_best = copy(x)
    f_best = f_val

    push!(loss_history, f_val)
    push!(grad_norm_history, norm(grad))
    push!(loss_data_history, terms.data)
    push!(loss_reg_eps_history, terms.reg_eps)
    push!(loss_reg_sigma_history, terms.reg_sigma)
    push!(step_alpha_history, NaN)  # iteration 0 has no line-search step
    push!(line_search_backtracks, 0)

    if verbose
        @printf("  iter %3d: loss = %.6e, |grad| = %.6e\n", 0, f_val, norm(grad))
        flush(stdout)
    end

    for iter in 1:max_iter
        # L-BFGS two-loop recursion
        q = copy(grad)
        alphas = Float64[]
        k = length(s_hist)

        for ii in k:-1:1
            rho = 1.0 / dot(y_hist[ii], s_hist[ii])
            alpha_i = rho * dot(s_hist[ii], q)
            push!(alphas, alpha_i)
            q .-= alpha_i .* y_hist[ii]
        end
        reverse!(alphas)

        # Initial Hessian scaling
        if k > 0
            gamma = dot(s_hist[end], y_hist[end]) / dot(y_hist[end], y_hist[end])
            r = gamma .* q
        else
            r = q
        end

        for ii in 1:k
            rho = 1.0 / dot(y_hist[ii], s_hist[ii])
            beta = rho * dot(y_hist[ii], r)
            r .+= (alphas[ii] - beta) .* s_hist[ii]
        end

        direction = -r

        # Verify descent direction; fall back to steepest descent if not
        dg = dot(grad, direction)
        if dg >= 0.0
            direction = -grad
            dg = -dot(grad, grad)
            empty!(s_hist)
            empty!(y_hist)
            if verbose
                @printf("  [L-BFGS reset: non-descent direction]\n")
                flush(stdout)
            end
        end

        # Backtracking line search (Armijo) with bounds projection
        alpha = step_size
        c1_armijo = 1e-4
        x_trial = clamp.(x .+ alpha .* direction, lb_vec, ub_vec)
        terms_new = objective_terms(x_trial)
        f_new = terms_new.total
        n_backtracks = 0

        ls_success = false
        for _ in 1:20
            if f_new <= f_val + c1_armijo * alpha * dg
                ls_success = true
                break
            end
            alpha *= 0.5
            n_backtracks += 1
            x_trial = clamp.(x .+ alpha .* direction, lb_vec, ub_vec)
            terms_new = objective_terms(x_trial)
            f_new = terms_new.total
        end

        # If line search failed, reject step and reset L-BFGS memory
        if !ls_success && f_new > f_val
            if verbose
                @printf("  [Line search failed, resetting L-BFGS]\n")
                flush(stdout)
            end
            empty!(s_hist)
            empty!(y_hist)
            # Small steepest-descent step as fallback
            gnorm = norm(grad)
            alpha = min(1e-2, 1.0 / gnorm)
            direction = -grad
            x_trial = clamp.(x .+ alpha .* direction, lb_vec, ub_vec)
            terms_new = objective_terms(x_trial)
            f_new = terms_new.total
        end

        # Update (x_trial already projected onto bounds)
        x_new = x_trial
        grad_new = multi_gradient(x_new)

        # Store L-BFGS pairs
        s_k = x_new .- x
        y_k = grad_new .- grad
        if dot(s_k, y_k) > 1e-20
            push!(s_hist, s_k)
            push!(y_hist, y_k)
            if length(s_hist) > m
                popfirst!(s_hist)
                popfirst!(y_hist)
            end
        end

        x = x_new
        grad = grad_new
        f_val = f_new

        # Update best iterate
        if f_val < f_best
            x_best = copy(x)
            f_best = f_val
        end

        push!(loss_history, f_val)
        push!(grad_norm_history, norm(grad))
        push!(loss_data_history, terms_new.data)
        push!(loss_reg_eps_history, terms_new.reg_eps)
        push!(loss_reg_sigma_history, terms_new.reg_sigma)
        push!(step_alpha_history, alpha)
        push!(line_search_backtracks, n_backtracks)

        if verbose
            @printf("  iter %3d: loss = %.6e, |grad| = %.6e, α = %.2e\n",
                    iter, f_val, norm(grad), alpha)
            flush(stdout)
        end

        # Convergence check
        if norm(grad) < 1e-10 * max(abs(f_val), 1.0)
            verbose && println("  Converged (gradient norm).")
            break
        end
    end

    if verbose && f_best < f_val
        @printf("  Using best iterate (loss = %.6e vs final %.6e)\n", f_best, f_val)
        flush(stdout)
    end

    # Unpack best parameters (not necessarily the final iterate)
    x_final = f_best < f_val ? x_best : x
    eps_inf_est = copy(eps_inf_init)
    sigma_est = copy(sigma_init)
    idx = 1
    for j in 1:ny, i in 1:nx
        if param_mask[i, j]
            if param_type == :eps_inf || param_type == :both
                eps_inf_est[i, j] = x_final[idx]
                idx += 1
            end
            if param_type == :sigma || param_type == :both
                sigma_est[i, j] = x_final[idx]
                idx += 1
            end
        end
    end

    return FWIResult(eps_inf_est, deps_map, sigma_est, loss_history,
                     grad_norm_history, loss_data_history,
                     loss_reg_eps_history, loss_reg_sigma_history,
                     step_alpha_history, line_search_backtracks,
                     length(loss_history) - 1)
end
