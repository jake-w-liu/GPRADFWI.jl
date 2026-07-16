@testset verbose = true "Inversion" begin

    small_config(; nx=20, ny=20, npml=4, nt=40) = create_config(
        nx=nx, ny=ny, dx=0.01, fc=300e6, npml=npml,
        src_ix=nx ÷ 2, src_iy=ny ÷ 2,
        rx_iy=npml + 2,
        rx_ix_list=collect((npml + 2):3:(nx - npml - 1)),
        nt=nt,
    )

    # ── Helper: small test problem matching the paper ─────────────────────

    function setup_small_inversion(; nt=100)
        nx, ny = 40, 30
        dx = 0.01
        fc = 300e6
        npml = 5

        config = create_config(
            nx=nx, ny=ny, dx=dx, fc=fc, npml=npml,
            src_ix=nx÷2, src_iy=ny÷2,
            rx_iy=npml+2,
            rx_ix_list=collect((npml+2):3:(nx-npml-1)),
            nt=nt
        )

        # Background: uniform ε∞=3, non-dispersive
        eps_inf_bg = fill(3.0, nx, ny)
        deps = zeros(nx, ny)
        tau = zeros(nx, ny)
        sigma = fill(0.001, nx, ny)

        # True model: anomaly with ε∞=5 in a 3×3 patch at center
        eps_inf_true = copy(eps_inf_bg)
        cx, cy = nx÷2, ny÷2
        for j in (cy-1):(cy+1), i in (cx-1):(cx+1)
            eps_inf_true[i, j] = 5.0
        end

        # Inversion mask: 7×7 region around anomaly
        param_mask = falses(nx, ny)
        for j in (cy-3):(cy+3), i in (cx-3):(cx+3)
            param_mask[i, j] = true
        end

        src = create_source(config)

        # Generate observed data from true model
        obs_data = run_forward!(config, eps_inf_true, deps, tau, sigma, src)

        return config, eps_inf_bg, eps_inf_true, deps, tau, sigma,
               param_mask, src, obs_data
    end

    # ── Category A: Finite-difference gradient ────────────────────────────

    @testset "fd_gradient — quadratic function" begin
        # Test on f(x) = 0.5 * ||x||², gradient = x
        f_quad(x) = 0.5 * dot(x, x)
        x = [1.0, -2.0, 3.0, 0.5]
        g = fd_gradient(f_quad, x)

        # Central FD of quadratic is exact to machine precision
        # Error bound: O(h²) for central FD; for quadratic, higher-order terms vanish
        # → limited only by Float64 round-off
        @test isapprox(g, x; rtol=1e-9)  # rtol=1e-9: h=1e-5, round-off ≈ eps/h ≈ 1e-11
    end

    @testset "fd_gradient — Rosenbrock function" begin
        # f(x,y) = (1-x)² + 100(y-x²)²
        # ∇f = [-2(1-x) - 400x(y-x²), 200(y-x²)]
        f_rosen(v) = (1.0 - v[1])^2 + 100.0 * (v[2] - v[1]^2)^2
        x = [1.5, 2.0]
        g = fd_gradient(f_rosen, x)

        grad_exact = [
            -2.0 * (1.0 - x[1]) - 400.0 * x[1] * (x[2] - x[1]^2),
            200.0 * (x[2] - x[1]^2)
        ]

        # rtol=1e-6: central FD with h=1e-5 on polynomial → O(h²)=O(1e-10),
        # but function values ~O(1) so absolute error dominates at ~1e-10 level;
        # relative error on gradient components ~1e-8 to 1e-6
        @test isapprox(g, grad_exact; rtol=1e-6)
    end

    # ── Category F: AD vs FD cross-validation ─────────────────────────────

    @testset "AD gradient matches FD gradient — quadratic" begin
        f_quad(x) = 0.5 * dot(x, x)
        x = [1.0, -2.0, 3.0]

        g_fd = fd_gradient(f_quad, x)
        g_ad = ad_gradient(f_quad, x)

        # Both should match the analytical gradient x
        # rtol=1e-9: FD limited by h², AD exact up to Float64
        @test isapprox(g_ad, x; rtol=1e-12)    # AD of quadratic is exact
        @test isapprox(g_ad, g_fd; rtol=1e-8)  # AD vs FD agree
    end

@testset "Forward-misfit parameter layout validation" begin
    config = small_config(nx=20, ny=20, npml=4, nt=4)
    eps_inf = fill(4.0, config.nx, config.ny)
    deps = zeros(config.nx, config.ny)
    tau = fill(1e-9, config.nx, config.ny)
    sigma = zeros(config.nx, config.ny)
    src = zeros(config.nt)
    obs = zeros(config.nt, length(config.rx_ix))
    mask = falses(config.nx, config.ny)
    mask[10, 10] = true

    args = (config, obs, src, eps_inf, deps, tau, sigma, mask)
    @test_throws ArgumentError forward_misfit([4.0], args..., :unknown)
    @test_throws DimensionMismatch forward_misfit(Float64[], args..., :eps_inf)
    @test_throws DimensionMismatch forward_misfit([4.0, 5.0], args..., :eps_inf)
    @test_throws DimensionMismatch forward_misfit([4.0], config, obs, src,
                                                   eps_inf, deps, tau[1:end-1, :],
                                                   sigma, mask, :eps_inf)
    @test_throws DimensionMismatch forward_misfit([4.0], config, obs, src,
                                                   eps_inf, deps, tau,
                                                   sigma[1:end-1, :], mask, :eps_inf)
end

@testset "Sigma-only regularization diagnostics" begin
    config = small_config(nx=20, ny=20, npml=4, nt=4)
    eps_inf = fill(4.0, config.nx, config.ny)
    deps = zeros(config.nx, config.ny)
    tau = fill(1e-9, config.nx, config.ny)
    sigma = zeros(config.nx, config.ny)
    sigma[10, 10] = 1e-3
    sigma[11, 10] = 3e-3
    src = zeros(config.nt)
    obs = zeros(config.nt, length(config.rx_ix))
    mask = falses(config.nx, config.ny)
    mask[10:11, 10] .= true

    callback_records = NamedTuple[]
    callback = state -> push!(callback_records, state)
    result = run_fwi_multisource([config], [obs], [src], eps_inf, deps, tau,
                                 sigma, mask; max_iter=1, param_type=:sigma,
                                 use_ad=false, verbose=false, lambda=0.0,
                                 lambda_sigma=1.0, callback=callback)
    expected = (sigma[11, 10] - sigma[10, 10])^2
    @test result.loss_reg_eps_history[1] == 0.0
    @test result.loss_reg_sigma_history[1] ≈ expected
    @test result.loss_history[1] ≈ result.loss_reg_sigma_history[1]
    @test length(callback_records) == 2
    @test callback_records[1].iteration == 0
    @test callback_records[2].iteration == 1
    @test callback_records[end].loss_total ≈ result.loss_history[end]
    @test all(0.0 .<= callback_records[end].params .<= 0.1)
end

@testset "Returned FWI model matches final history entry" begin
    config = small_config(nx=20, ny=20, npml=4, nt=40)
    eps_init = fill(4.0, config.nx, config.ny)
    eps_true = copy(eps_init)
    eps_true[config.source.ix, config.source.iy] = 6.0
    deps = zeros(config.nx, config.ny)
    tau = fill(1e-9, config.nx, config.ny)
    sigma = zeros(config.nx, config.ny)
    src = zeros(config.nt)
    src[2] = 1e-9
    obs = run_forward!(config, eps_true, deps, tau, sigma, src)
    mask = falses(config.nx, config.ny)
    mask[config.source.ix, config.source.iy] = true

    result = run_fwi(config, obs, src, eps_init, deps, tau, sigma, mask;
                     max_iter=1, param_type=:eps_inf, use_ad=false,
                     verbose=false)
    final_params = [result.eps_inf_est[config.source.ix, config.source.iy]]
    final_loss = forward_misfit(final_params, config, obs, src, eps_init,
                                deps, tau, sigma, mask, :eps_inf)
    @test final_loss ≈ result.loss_history[end] rtol=1e-12 atol=1e-12
end

@testset "AD gradient matches FD gradient — FDTD misfit" begin
        Random.seed!(401)
        config, eps_inf_bg, eps_inf_true, deps, tau, sigma,
            param_mask, src, obs_data = setup_small_inversion(nt=50)

        # Pack initial parameters
        np = count(param_mask)
        x0 = Float64[]
        for j in 1:config.ny, i in 1:config.nx
            if param_mask[i, j]
                push!(x0, eps_inf_bg[i, j])
            end
        end

        objective(x) = forward_misfit(x, config, obs_data, src,
                                       eps_inf_bg, deps, tau, sigma,
                                       param_mask, :eps_inf)

        g_fd = fd_gradient(objective, x0; h=1e-5)

        # Use direct Enzyme.autodiff on forward_misfit (not closures)
        dx = zeros(length(x0))
        Enzyme.autodiff(Enzyme.Reverse, forward_misfit, Enzyme.Active,
            Enzyme.Duplicated(x0, dx),
            Enzyme.Const(config),
            Enzyme.Const(obs_data),
            Enzyme.Const(src),
            Enzyme.Const(eps_inf_bg),
            Enzyme.Const(deps),
            Enzyme.Const(tau),
            Enzyme.Const(sigma),
            Enzyme.Const(param_mask),
            Enzyme.Const(:eps_inf))

        # The AD and FD gradients should agree to within FD truncation error
        # Central FD: error ~ O(h²) ≈ 1e-10, relative to gradient norm
        rel_err = norm(dx - g_fd) / max(norm(g_fd), 1e-20)
        @test rel_err < 1e-5  # conservative: allow for FDTD accumulation through nt=50 steps
    end

    # ── Category A: forward_misfit zero at true model ─────────────────────

    @testset "forward_misfit is zero at true model" begin
        Random.seed!(402)
        config, eps_inf_bg, eps_inf_true, deps, tau, sigma,
            param_mask, src, obs_data = setup_small_inversion(nt=50)

        # Pack TRUE parameters
        x_true = Float64[]
        for j in 1:config.ny, i in 1:config.nx
            if param_mask[i, j]
                push!(x_true, eps_inf_true[i, j])
            end
        end

        misfit = forward_misfit(x_true, config, obs_data, src,
                                 eps_inf_bg, deps, tau, sigma,
                                 param_mask, :eps_inf)

        # Misfit at true model should be exactly zero (synthetic data = forward(true))
        @test isapprox(misfit, 0.0; atol=1e-10)  # atol=1e-10: Float64 accumulation over nt×nrx
    end

    @testset "forward_misfit is positive at wrong model" begin
        Random.seed!(403)
        config, eps_inf_bg, eps_inf_true, deps, tau, sigma,
            param_mask, src, obs_data = setup_small_inversion(nt=50)

        # Pack BACKGROUND (wrong) parameters
        x_wrong = Float64[]
        for j in 1:config.ny, i in 1:config.nx
            if param_mask[i, j]
                push!(x_wrong, eps_inf_bg[i, j])
            end
        end

        misfit = forward_misfit(x_wrong, config, obs_data, src,
                                 eps_inf_bg, deps, tau, sigma,
                                 param_mask, :eps_inf)

        @test misfit > 0.0  # wrong model must have positive misfit
    end

    # ── Category F: FWI round-trip — recover known anomaly ────────────────

    @testset "FWI round-trip: recover anomaly" begin
        Random.seed!(404)
        config, eps_inf_bg, eps_inf_true, deps, tau, sigma,
            param_mask, src, obs_data = setup_small_inversion(nt=80)

        # Run FWI from background model
        result = run_fwi(config, obs_data, src,
                         eps_inf_bg, deps, tau, sigma, param_mask;
                         max_iter=15, param_type=:eps_inf,
                         use_ad=false,  # use FD for reliability in tests
                         verbose=false)

        # Loss should decrease
        @test result.loss_history[end] < result.loss_history[1]  # FWI must reduce misfit

        # Loss reduction should be significant (> 50%)
        reduction = 1.0 - result.loss_history[end] / result.loss_history[1]
        @test reduction > 0.5  # at least 50% misfit reduction

        # The reconstructed ε∞ at the anomaly center should be > background
        cx, cy = config.nx ÷ 2, config.ny ÷ 2
        @test result.eps_inf_est[cx, cy] > eps_inf_bg[cx, cy]  # anomaly detected (εr > 3)
    end

    # ── Category B: Gradient descent direction ────────────────────────────

    @testset "Gradient points in descent direction" begin
        Random.seed!(405)
        config, eps_inf_bg, eps_inf_true, deps, tau, sigma,
            param_mask, src, obs_data = setup_small_inversion(nt=50)

        x0 = Float64[]
        for j in 1:config.ny, i in 1:config.nx
            if param_mask[i, j]
                push!(x0, eps_inf_bg[i, j])
            end
        end

        objective(x) = forward_misfit(x, config, obs_data, src,
                                       eps_inf_bg, deps, tau, sigma,
                                       param_mask, :eps_inf)

        f0 = objective(x0)
        g = fd_gradient(objective, x0)

        # Step in negative gradient direction should decrease the objective
        alpha = 1e-6 / norm(g)  # small step to stay in linear regime
        x_step = x0 .- alpha .* g
        f_step = objective(x_step)

        @test f_step < f0  # steepest descent must decrease cost
    end

    @testset "Armijo failure triggers manuscript fallback" begin
        ls_success = false
        f_val = 1.0
        f_new = 0.99999999995
        armijo_rhs = 0.99999999990

        @test f_new < f_val
        @test f_new > armijo_rhs
        @test GPRADFWI._line_search_needs_fallback(ls_success)
    end

    # ── Category D: Single-cell inversion ─────────────────────────────────

    @testset "Single-parameter inversion" begin
        Random.seed!(406)
        nx, ny = 30, 25
        dx = 0.01
        fc = 300e6
        npml = 5
        nt = 60

        config = create_config(
            nx=nx, ny=ny, dx=dx, fc=fc, npml=npml,
            src_ix=15, src_iy=13, rx_iy=npml+2,
            rx_ix_list=[10, 15, 20], nt=nt
        )

        # True model: single cell perturbation
        eps_inf_true = fill(3.0, nx, ny)
        eps_inf_true[15, 13] = 5.0
        deps = zeros(nx, ny)
        tau = zeros(nx, ny)
        sigma = fill(0.001, nx, ny)

        eps_inf_bg = fill(3.0, nx, ny)
        param_mask = falses(nx, ny)
        param_mask[15, 13] = true  # single cell

        src = create_source(config)
        obs_data = run_forward!(config, eps_inf_true, deps, tau, sigma, src)

        result = run_fwi(config, obs_data, src,
                         eps_inf_bg, deps, tau, sigma, param_mask;
                         max_iter=10, param_type=:eps_inf,
                         use_ad=false, verbose=false)

        # The single inverted parameter should move toward true value (5.0)
        @test result.eps_inf_est[15, 13] > 3.5  # moved from 3.0 toward 5.0
    end
end
