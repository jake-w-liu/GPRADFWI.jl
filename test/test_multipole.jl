using Test
using GPRADFWI
using Enzyme
using LinearAlgebra

if !isdefined(GPRADFWI, :MultiPoleDebyeMedium)
    Base.include(GPRADFWI, joinpath(@__DIR__, "..", "src", "multipole.jl"))
end

const MP = GPRADFWI

@testset verbose = true "Multi-pole Debye FDTD" begin
    function multipole_test_config(; nx=30, ny=24, nt=100)
        return create_config(
            nx=nx, ny=ny, dx=0.01, fc=300e6, npml=4,
            src_ix=nx ÷ 2, src_iy=ny ÷ 2,
            rx_iy=ny ÷ 2,
            rx_ix_list=[nx ÷ 2 - 5, nx ÷ 2 + 5],
            nt=nt,
        )
    end

    @testset "one-pole trace equivalence" begin
        config = multipole_test_config(nt=300)
        nx, ny = config.nx, config.ny
        eps_inf = fill(4.0, nx, ny)
        deps = fill(4.0, nx, ny)
        tau = fill(0.3e-9, nx, ny)
        sigma = fill(0.005, nx, ny)
        source = create_source(config)

        reference = run_forward!(config, eps_inf, deps, tau, sigma, source)
        medium = MP.MultiPoleDebyeMedium(reshape(copy(deps), nx, ny, 1),
                                         reshape(copy(tau), nx, ny, 1))
        candidate = MP.run_forward_multipole!(config, eps_inf, medium, sigma, source)

        @test isapprox(candidate, reference; rtol=1e-12, atol=1e-12)
    end

    @testset "zero-strength second pole equivalence" begin
        config = multipole_test_config(nt=300)
        nx, ny = config.nx, config.ny
        eps_inf = fill(4.0, nx, ny)
        sigma = fill(0.005, nx, ny)
        source = create_source(config)

        deps_one = fill(4.0, nx, ny, 1)
        tau_one = fill(0.3e-9, nx, ny, 1)
        one_pole = MP.MultiPoleDebyeMedium(deps_one, tau_one)

        deps_two = cat(deps_one, zeros(nx, ny, 1); dims=3)
        tau_two = cat(tau_one, fill(1.2e-9, nx, ny, 1); dims=3)
        two_pole = MP.MultiPoleDebyeMedium(deps_two, tau_two)

        trace_one = MP.run_forward_multipole!(config, eps_inf, one_pole, sigma, source)
        trace_two = MP.run_forward_multipole!(config, eps_inf, two_pole, sigma, source)
        @test trace_two == trace_one
    end

    @testset "two-pole constitutive frequency oracle" begin
        deps = [2.5, 1.5]
        tau = [0.15e-9, 1.2e-9]
        dt = 0.1e-12
        frequencies = [50e6, 100e6, 250e6, 500e6, 1e9, 2e9]

        for frequency in frequencies
            omega = 2.0 * pi * frequency
            analytic = sum(deps[p] / (1.0 + im * omega * tau[p]) for p in eachindex(deps))
            discrete = MP.discrete_debye_susceptibility(deps, tau, omega, dt)
            relative_error = abs(discrete - analytic) / abs(analytic)
            @test relative_error < 2e-3
        end
    end

    @testset "two-pole full FDTD remains finite" begin
        config = multipole_test_config(nt=300)
        nx, ny = config.nx, config.ny
        eps_inf = fill(4.0, nx, ny)
        sigma = fill(0.005, nx, ny)
        deps = Array{Float64,3}(undef, nx, ny, 2)
        tau = Array{Float64,3}(undef, nx, ny, 2)
        deps[:, :, 1] .= 2.5
        deps[:, :, 2] .= 1.5
        tau[:, :, 1] .= 0.15e-9
        tau[:, :, 2] .= 1.2e-9
        medium = MP.MultiPoleDebyeMedium(deps, tau)

        trace = MP.run_forward_multipole!(config, eps_inf, medium, sigma, create_source(config))
        @test all(isfinite, trace)
        @test maximum(abs, trace) > 0.0
    end

    @testset "Enzyme directional derivative matches central FD" begin
        config = multipole_test_config(nx=24, ny=20, nt=240)
        nx, ny = config.nx, config.ny
        eps_bg = fill(4.0, nx, ny)
        eps_true = copy(eps_bg)
        sigma = fill(0.005, nx, ny)
        deps = Array{Float64,3}(undef, nx, ny, 2)
        tau = Array{Float64,3}(undef, nx, ny, 2)
        deps[:, :, 1] .= 2.5
        deps[:, :, 2] .= 1.5
        tau[:, :, 1] .= 0.15e-9
        tau[:, :, 2] .= 1.2e-9
        medium = MP.MultiPoleDebyeMedium(deps, tau)

        cx, cy = nx ÷ 2, ny ÷ 2
        mask = falses(nx, ny)
        for j in cy-1:cy+1, i in cx-1:cx+1
            mask[i, j] = true
            eps_true[i, j] = 5.0
        end

        source = create_source(config)
        observed = MP.run_forward_multipole!(config, eps_true, medium, sigma, source)
        x0 = fill(4.0, count(mask))
        direction = [isodd(k) ? 1.0 : -1.0 for k in eachindex(x0)]

        gradient = zeros(length(x0))
        Enzyme.autodiff(
            Enzyme.Reverse,
            MP.multipole_forward_misfit_eps,
            Enzyme.Active,
            Enzyme.Duplicated(x0, gradient),
            Enzyme.Const(config),
            Enzyme.Const(observed),
            Enzyme.Const(source),
            Enzyme.Const(eps_bg),
            Enzyme.Const(medium),
            Enzyme.Const(sigma),
            Enzyme.Const(mask),
        )

        h = 3e-5
        f_plus = MP.multipole_forward_misfit_eps(
            x0 .+ h .* direction, config, observed, source, eps_bg, medium, sigma, mask)
        f_minus = MP.multipole_forward_misfit_eps(
            x0 .- h .* direction, config, observed, source, eps_bg, medium, sigma, mask)
        fd_directional = (f_plus - f_minus) / (2.0 * h)
        ad_directional = dot(gradient, direction)
        relative_error = abs(ad_directional - fd_directional) /
                         max(abs(ad_directional), abs(fd_directional), 1e-20)
        @test relative_error < 2e-4
    end
end
