@testset verbose = true "Debye Coefficients" begin

    init_debye = GPRADFWI.init_debye_coeffs

    # ── Category A: Hand-computed coefficients ────────────────────────────

    @testset "Single dispersive cell — hand computation" begin
        # Setup: one cell, known parameters
        ei = 4.0      # ε∞
        de = 4.0      # Δε
        ta = 0.3e-9   # τ = 0.3 ns
        sg = 0.005    # σ = 5 mS/m
        dt = 11.68e-12 # dt ≈ 11.68 ps

        eps_inf = fill(ei, 1, 1)
        deps = fill(de, 1, 1)
        tau = fill(ta, 1, 1)
        sigma = fill(sg, 1, 1)

        dc = init_debye(eps_inf, deps, tau, sigma, dt, 1, 1)

        # Hand calculation:
        # denom_p = 2τ + dt = 2×0.3e-9 + 11.68e-12 = 6.1168e-10
        denom_p = 2.0 * ta + dt
        c1_expected = (2.0 * ta - dt) / denom_p
        c2_expected = 2.0 * GPRADFWI.eps0 * de * dt / denom_p

        # atol=1e-15: direct Float64 arithmetic, no iteration
        @test isapprox(dc.c1[1, 1], c1_expected; atol=1e-15)
        @test isapprox(dc.c2[1, 1], c2_expected; atol=1e-15)

        # Combined E-field coefficients:
        # D = ε₀·ε∞ + c2 + σ·dt/2
        D = GPRADFWI.eps0 * ei + c2_expected + sg * dt / 2.0
        ca_expected = (GPRADFWI.eps0 * ei - sg * dt / 2.0) / D
        cb_expected = dt / D
        cp_expected = (1.0 - c1_expected) / D

        @test isapprox(dc.ca[1, 1], ca_expected; atol=1e-15)
        @test isapprox(dc.cb[1, 1], cb_expected; atol=1e-15)
        @test isapprox(dc.cp[1, 1], cp_expected; atol=1e-15)
    end

    @testset "Non-dispersive cell — hand computation" begin
        # When Δε=0 and τ=0, should recover standard lossy dielectric update
        ei = 4.0
        sg = 0.005
        dt = 11.68e-12

        eps_inf = fill(ei, 1, 1)
        deps = fill(0.0, 1, 1)
        tau = fill(0.0, 1, 1)
        sigma = fill(sg, 1, 1)

        dc = init_debye(eps_inf, deps, tau, sigma, dt, 1, 1)

        # Non-dispersive: c1=0, c2=0, cp=0
        @test dc.c1[1, 1] == 0.0  # exact zero for non-dispersive
        @test dc.c2[1, 1] == 0.0
        @test dc.cp[1, 1] == 0.0

        # ca = (ε₀ε∞ - σdt/2) / (ε₀ε∞ + σdt/2)
        D = GPRADFWI.eps0 * ei + sg * dt / 2.0
        ca_expected = (GPRADFWI.eps0 * ei - sg * dt / 2.0) / D
        cb_expected = dt / D

        @test isapprox(dc.ca[1, 1], ca_expected; atol=1e-15)
        @test isapprox(dc.cb[1, 1], cb_expected; atol=1e-15)
    end

    # ── Category B: Mathematical properties ───────────────────────────────

    @testset "Coefficient bounds — physical constraints" begin
        Random.seed!(201)
        dt = 11.68e-12

        for _ in 1:10
            # Random physically plausible Debye parameters
            ei = 1.0 + rand() * 20.0          # ε∞ ∈ [1, 21]
            de = rand() * 15.0                 # Δε ∈ [0, 15]
            ta = (0.01 + rand() * 10.0) * 1e-9 # τ ∈ [0.01, 10.01] ns
            sg = rand() * 0.1                  # σ ∈ [0, 0.1] S/m

            eps_inf = fill(ei, 1, 1)
            deps = fill(de, 1, 1)
            tau = fill(ta, 1, 1)
            sigma = fill(sg, 1, 1)

            dc = init_debye(eps_inf, deps, tau, sigma, dt, 1, 1)

            # c1 ∈ (-1, 1): exponential decay factor for polarization
            # c1 = (2τ-dt)/(2τ+dt). Since τ > 0 and dt > 0: -1 < c1 < 1
            @test -1.0 < dc.c1[1, 1] < 1.0  # stability bound for ADE

            # c2 ≥ 0: coupling from E to P (non-negative for physical media)
            @test dc.c2[1, 1] >= 0.0  # Δε ≥ 0 ensures non-negative coupling

            # ca ∈ (0, 1]: dissipation means ca < 1, but ca > 0 for stable update
            @test 0.0 < dc.ca[1, 1] <= 1.0  # must be positive and ≤ 1

            # cb > 0: time step contribution must be positive
            @test dc.cb[1, 1] > 0.0

            # cp ≥ 0: polarization feedback
            @test dc.cp[1, 1] >= 0.0  # (1-c1) ≥ 0 since c1 < 1
        end
    end

    @testset "Lossless non-dispersive reduces to vacuum update" begin
        # ε∞=1, σ=0, Δε=0 → ca=1, cb=dt/(ε₀), cp=0
        dt = 1e-11
        eps_inf = fill(1.0, 1, 1)
        deps = fill(0.0, 1, 1)
        tau = fill(0.0, 1, 1)
        sigma = fill(0.0, 1, 1)

        dc = init_debye(eps_inf, deps, tau, sigma, dt, 1, 1)

        @test isapprox(dc.ca[1, 1], 1.0; atol=1e-15)  # no loss → ca = 1 exactly
        @test isapprox(dc.cb[1, 1], dt / GPRADFWI.eps0; rtol=1e-14)  # cb = dt/ε₀
        @test dc.cp[1, 1] == 0.0  # no dispersion
    end

    # ── Category D: Edge cases ────────────────────────────────────────────

    @testset "Very large permittivity" begin
        # ε∞ = 80 (water): should not overflow or produce NaN
        dt = 1e-11
        eps_inf = fill(80.0, 1, 1)
        deps = fill(5.0, 1, 1)
        tau = fill(8.0e-12, 1, 1)
        sigma = fill(0.01, 1, 1)

        dc = init_debye(eps_inf, deps, tau, sigma, dt, 1, 1)

        @test isfinite(dc.ca[1, 1])  # must not overflow
        @test isfinite(dc.cb[1, 1])
        @test isfinite(dc.cp[1, 1])
        @test isfinite(dc.c1[1, 1])
        @test isfinite(dc.c2[1, 1])
    end

    # ── Category E: Error handling ────────────────────────────────────────

    @testset "Dimension assertions" begin
        dt = 1e-11
        # Wrong-sized arrays should trigger assertion
        @test_throws AssertionError init_debye(
            fill(1.0, 3, 3), fill(0.0, 2, 3), fill(0.0, 3, 3), fill(0.0, 3, 3),
            dt, 3, 3
        )
        # Negative dt should trigger assertion
        @test_throws AssertionError init_debye(
            fill(1.0, 2, 2), fill(0.0, 2, 2), fill(0.0, 2, 2), fill(0.0, 2, 2),
            -1e-11, 2, 2
        )
    end

    # ── Category C: Static permittivity limit ─────────────────────────────

    @testset "DC limit: Δε contributes to static permittivity" begin
        # At ω→0: ε(0) = ε∞ + Δε = εs
        # The Debye ADE at DC (steady state, ∂P/∂t=0): P = ε₀·Δε·E
        # So total D = ε₀·ε∞·E + P = ε₀·(ε∞+Δε)·E = ε₀·εs·E
        # After many time steps with constant E, P should converge to ε₀·Δε·E
        ei = 4.0
        de = 4.0  # εs = 8
        ta = 0.3e-9
        dt = 11.68e-12

        dc = GPRADFWI.init_debye_coeffs(fill(ei, 1, 1), fill(de, 1, 1),
                                          fill(ta, 1, 1), fill(0.0, 1, 1), dt, 1, 1)

        # Iterate ADE with constant E=1.0 until P converges
        P = 0.0
        E = 1.0
        for _ in 1:100_000  # many iterations >> τ/dt ≈ 26
            P = dc.c1[1, 1] * P + dc.c2[1, 1] * E
        end
        P_expected = GPRADFWI.eps0 * de * E  # steady-state: P = ε₀·Δε·E

        # rtol=1e-6: iterative convergence, limited by (c1)^N residual
        # c1 ≈ 0.981, after 100k iters residual ≈ c1^100000 ≈ 10^{-830} → effectively exact
        @test isapprox(P, P_expected; rtol=1e-12)
    end
end
