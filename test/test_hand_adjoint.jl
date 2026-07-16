using Test
using Random
using LinearAlgebra
using Enzyme
using GPRADFWI

if !isdefined(GPRADFWI, :HandAdjointTape)
    Base.include(GPRADFWI, joinpath(@__DIR__, "..", "src", "hand_adjoint.jl"))
end

function _matched_hand_problem()
    nx, ny, p, nt = 18, 16, 3, 38
    config = create_config(
        nx=nx, ny=ny, dx=0.01, fc=600e6, npml=p,
        src_ix=9, src_iy=7, rx_iy=5, rx_ix_list=[4, 9, 14], nt=nt,
    )
    src = zeros(nt)
    src[2] = 1.0e-3
    src[3] = -6.0e-4
    src[4] = 2.0e-4

    eps_bg = fill(3.2, nx, ny)
    deps = fill(1.4, nx, ny)
    tau = fill(0.35e-9, nx, ny)
    sigma = fill(0.002, nx, ny)

    mask = falses(nx, ny)
    mask[9, 8] = true
    mask[10, 8] = true
    mask[nx - p, 8] = true
    mask[nx - p + 1, 8] = true
    mask[9, ny - p] = true
    mask[9, ny - p + 1] = true

    eps_true = copy(eps_bg)
    increments = (0.7, 0.4, 0.3, 0.5, 0.25, 0.45)
    k = 0
    for j in 1:ny, i in 1:nx
        if mask[i, j]
            k += 1
            eps_true[i, j] += increments[k]
        end
    end
    obs = run_forward!(config, eps_true, deps, tau, sigma, src)
    x0 = [eps_bg[i, j] for j in 1:ny for i in 1:nx if mask[i, j]]
    return (; config, src, eps_bg, deps, tau, sigma, mask, obs, x0)
end

@testset "Full Debye+CPML hand adjoint" begin
    @testset "source updates polarization consistently" begin
        config = create_config(
            nx=10, ny=9, dx=0.01, fc=600e6, npml=2,
            src_ix=5, src_iy=5, rx_iy=5, rx_ix_list=[5], nt=1,
        )
        eps_inf = fill(3.0, 10, 9)
        deps = fill(1.2, 10, 9)
        tau = fill(0.4e-9, 10, 9)
        sigma = fill(0.001, 10, 9)
        src = [2.0e-4]
        rec, tape = GPRADFWI.run_forward_hand_tape(
            config, eps_inf, deps, tau, sigma, src,
        )
        dc = GPRADFWI.init_debye_coeffs(
            eps_inf, deps, tau, sigma, config.dt, config.nx, config.ny,
        )
        delta_e = dc.cb[5, 5] * src[1] / (config.dx * config.dy)
        @test isapprox(tape.Ez[5, 5, 2], delta_e; rtol=1e-14, atol=0.0)
        @test isapprox(tape.Pz[5, 5, 2], dc.c2[5, 5] * delta_e; rtol=1e-14, atol=0.0)
        @test rec[1, 1] == tape.Ez[5, 5, 2]
    end

    @testset "multistep H identity adjoint" begin
        config = create_config(
            nx=10, ny=9, dx=0.01, fc=600e6, npml=2,
            src_ix=5, src_iy=5, rx_iy=5, rx_ix_list=[5, 6], nt=3,
        )
        src = [1.0e-3, 0.0, 0.0]
        eps_inf = fill(3.0, 10, 9)
        deps = fill(1.2, 10, 9)
        tau = fill(0.4e-9, 10, 9)
        sigma = fill(0.001, 10, 9)
        mask = falses(10, 9)
        mask[5, 5] = true
        mask[6, 5] = true
        eps_true = copy(eps_inf)
        eps_true[5, 5] = 3.5
        eps_true[6, 5] = 3.4
        obs = run_forward!(config, eps_true, deps, tau, sigma, src)
        x0 = [3.0, 3.0]
        g_hand = GPRADFWI.hand_adjoint_gradient(
            x0, config, obs, src, eps_inf, deps, tau, sigma, mask,
        )
        objective(x) = forward_misfit(
            x, config, obs, src, eps_inf, deps, tau, sigma, mask, :eps_inf,
        )
        h = 1e-5
        g_fd = [
            (objective(x0 .+ h .* [1.0, 0.0]) - objective(x0 .- h .* [1.0, 0.0])) / (2h),
            (objective(x0 .+ h .* [0.0, 1.0]) - objective(x0 .- h .* [0.0, 1.0])) / (2h),
        ]
        @test norm(g_hand - g_fd) / norm(g_fd) < 2e-9
    end

    problem = _matched_hand_problem()

    @testset "matched corrected production forward" begin
        rec_prod = run_forward!(
            problem.config, problem.eps_bg, problem.deps, problem.tau,
            problem.sigma, problem.src,
        )
        rec_hand, tape = GPRADFWI.run_forward_hand_tape(
            problem.config, problem.eps_bg, problem.deps, problem.tau,
            problem.sigma, problem.src,
        )
        @test isapprox(rec_hand, rec_prod; rtol=2e-13, atol=1e-13)
        @test any(!iszero, tape.psi_hyx_x2)
        @test any(!iszero, tape.psi_hxy_y2)
        @test any(!iszero, tape.psi_ezx_x2)
        @test any(!iszero, tape.psi_ezy_y2)

        expected = sizeof(Float64) * (problem.config.nt + 1) *
                   (4 * problem.config.nx * problem.config.ny +
                    4 * problem.config.cpml.npml * problem.config.ny +
                    4 * problem.config.nx * problem.config.cpml.npml)
        @test GPRADFWI.hand_adjoint_store_all_bytes(problem.config) == expected
    end

    @testset "Enzyme and hand gradients agree" begin
        g_hand = GPRADFWI.hand_adjoint_gradient(
            problem.x0, problem.config, problem.obs, problem.src,
            problem.eps_bg, problem.deps, problem.tau, problem.sigma,
            problem.mask,
        )
        g_ad = zeros(length(problem.x0))
        Enzyme.autodiff(
            Enzyme.Reverse, GPRADFWI.forward_misfit, Enzyme.Active,
            Enzyme.Duplicated(problem.x0, g_ad),
            Enzyme.Const(problem.config), Enzyme.Const(problem.obs),
            Enzyme.Const(problem.src), Enzyme.Const(problem.eps_bg),
            Enzyme.Const(problem.deps), Enzyme.Const(problem.tau),
            Enzyme.Const(problem.sigma), Enzyme.Const(problem.mask),
            Enzyme.Const(:eps_inf),
        )
        rel_l2 = norm(g_hand - g_ad) / max(norm(g_ad), 1e-30)
        cosine = dot(g_hand, g_ad) / max(norm(g_hand) * norm(g_ad), 1e-30)
        @test rel_l2 < 1e-9
        @test cosine > 1.0 - 1e-11
        @test all(isfinite, g_hand)

        # The selected cells deliberately straddle the first right/top E-PML
        # nodes, so this comparison exercises the corrected staggered mapping.
        @test problem.mask[problem.config.nx - problem.config.cpml.npml + 1, 8]
        @test problem.mask[9, problem.config.ny - problem.config.cpml.npml + 1]
    end

    @testset "directional finite difference" begin
        Random.seed!(20260716)
        direction = randn(length(problem.x0))
        direction ./= norm(direction)
        g_hand = GPRADFWI.hand_adjoint_gradient(
            problem.x0, problem.config, problem.obs, problem.src,
            problem.eps_bg, problem.deps, problem.tau, problem.sigma,
            problem.mask,
        )
        objective(x) = GPRADFWI.forward_misfit(
            x, problem.config, problem.obs, problem.src, problem.eps_bg,
            problem.deps, problem.tau, problem.sigma, problem.mask, :eps_inf,
        )
        h = 1e-5
        fd = (objective(problem.x0 .+ h .* direction) -
              objective(problem.x0 .- h .* direction)) / (2.0 * h)
        adj = dot(g_hand, direction)
        rel = abs(adj - fd) / max(abs(adj), abs(fd), 1e-12)
        @test rel < 2e-4
    end
end
