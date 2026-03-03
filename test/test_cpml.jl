@testset verbose = true "CPML" begin

    # ── Category A: PML distance functions ────────────────────────────────

    @testset "PML distance — E-nodes" begin
        pml_dist_e = GPRADFWI._pml_distance_e
        n = 100
        npml = 10

        # At left boundary edge (i=1): distance = (10-1+1)/10 = 1.0
        @test pml_dist_e(1, n, npml) == 1.0  # outermost PML cell

        # At PML-interior interface (i=npml=10): distance = (10-10+1)/10 = 0.1
        @test pml_dist_e(npml, n, npml) == 1.0 / npml  # innermost PML cell

        # Just inside interior (i=npml+1=11): not in PML
        @test pml_dist_e(npml + 1, n, npml) == -1.0  # outside PML

        # At right boundary edge (i=n=100): distance = (100-(100-10))/10 = 1.0
        @test pml_dist_e(n, n, npml) == 1.0  # outermost right PML cell

        # Interior cell: should return -1.0
        @test pml_dist_e(50, n, npml) == -1.0  # mid-domain
    end

    @testset "PML distance — H-nodes (half-integer offset)" begin
        pml_dist_h = GPRADFWI._pml_distance_h
        n = 100
        npml = 10

        # At left boundary (i=1): distance = (10-1+0.5)/10 = 0.95
        @test pml_dist_h(1, n, npml) == 0.95  # half-cell offset from E-node

        # H-node distance should differ from E-node by 0.5/npml
        pml_dist_e = GPRADFWI._pml_distance_e
        for i in 1:npml
            de = pml_dist_e(i, n, npml)
            dh = pml_dist_h(i, n, npml)
            @test isapprox(de - dh, 0.5 / npml; atol=1e-15)  # half-cell shift
        end
    end

    # ── Category A: PML profile ───────────────────────────────────────────

    @testset "PML profile — polynomial grading" begin
        pml_profile = GPRADFWI._pml_profile
        npml = 10
        order = 3
        sigma_opt = 1.0  # normalized
        kappa_max = 11.0
        alpha_max = 0.05

        # At d=0 (interior boundary): σ=0, κ=1, α=α_max
        s0, k0, a0 = pml_profile(0.0, npml, order, sigma_opt, kappa_max, alpha_max)
        @test s0 == 0.0   # σ = σ_opt · 0^order = 0
        @test k0 == 1.0   # κ = 1 + (κ_max-1)·0^order = 1
        @test a0 == alpha_max  # α = α_max·(1-0) = α_max

        # At d=1 (outer boundary): σ=σ_opt, κ=κ_max, α=0
        s1, k1, a1 = pml_profile(1.0, npml, order, sigma_opt, kappa_max, alpha_max)
        @test s1 == sigma_opt  # σ = σ_opt · 1^order
        @test k1 == kappa_max  # κ = 1 + (κ_max-1)·1
        @test a1 == 0.0        # α = α_max·(1-1) = 0

        # At d=0.5: σ = σ_opt·0.125, κ = 1+10·0.125 = 2.25, α = 0.025
        s5, k5, a5 = pml_profile(0.5, npml, order, sigma_opt, kappa_max, alpha_max)
        @test isapprox(s5, sigma_opt * 0.5^order; atol=1e-15)
        @test isapprox(k5, 1.0 + (kappa_max - 1.0) * 0.5^order; atol=1e-15)
        @test isapprox(a5, alpha_max * 0.5; atol=1e-15)
    end

    # ── Category B: Monotonicity ──────────────────────────────────────────

    @testset "Profile monotonicity" begin
        pml_profile = GPRADFWI._pml_profile
        npml = 10
        order = 3
        sigma_opt = 1.0
        kappa_max = 11.0
        alpha_max = 0.05

        # σ and κ must increase from d=0 to d=1
        # α must decrease from d=0 to d=1
        ds = range(0, 1; length=50)
        sigmas = Float64[]
        kappas = Float64[]
        alphas = Float64[]
        for d in ds
            s, k, a = pml_profile(d, npml, order, sigma_opt, kappa_max, alpha_max)
            push!(sigmas, s)
            push!(kappas, k)
            push!(alphas, a)
        end

        # σ(d) monotonically non-decreasing
        @test all(diff(sigmas) .>= 0.0)  # polynomial grading: d^m is monotone
        # κ(d) monotonically non-decreasing
        @test all(diff(kappas) .>= 0.0)
        # α(d) monotonically non-increasing
        @test all(diff(alphas) .<= 0.0)
    end

    # ── Category A: CPML coefficient computation ──────────────────────────

    @testset "CPML coefficients b and c" begin
        # Verify b = exp(-(σ/κ + α)·dt/ε₀)
        # and    c = σ/(σκ + κ²α)·(b - 1)
        config = create_config(
            nx=40, ny=30, dx=0.01, fc=300e6, npml=5,
            src_ix=20, src_iy=10, rx_iy=5, rx_ix_list=[20],
            nt=10
        )
        cpml = GPRADFWI.init_cpml(config)

        # Check a PML cell on the left boundary (i=1)
        pml_profile = GPRADFWI._pml_profile
        pml_dist_e = GPRADFWI._pml_distance_e
        de = pml_dist_e(1, config.nx, config.cpml.npml)
        sigma_opt_x = config.cpml.sigma_fac * (config.cpml.order + 1) / (2.0 * eta0 * config.dx)
        se, ke, ae = pml_profile(de, config.cpml.npml, config.cpml.order,
                                  sigma_opt_x, config.cpml.kappa_max, config.cpml.alpha_max)

        b_expected = exp(-(se / ke + ae) * config.dt / eps0)
        # c = σ/(σκ + κ²α)·(b-1), handle division carefully
        c_expected = se / (se * ke + ke^2 * ae) * (b_expected - 1.0)

        @test isapprox(cpml.be_x[1], b_expected; rtol=1e-14)  # rtol=1e-14: exp well-conditioned
        @test isapprox(cpml.ce_x[1], c_expected; rtol=1e-12)  # rtol=1e-12: (b-1) near zero subtractive cancellation
    end

    # ── Category D: Interior cells have zero CPML ─────────────────────────

    @testset "Interior cells: zero CPML coefficients" begin
        config = create_config(
            nx=40, ny=30, dx=0.01, fc=300e6, npml=5,
            src_ix=20, src_iy=10, rx_iy=5, rx_ix_list=[20],
            nt=10
        )
        cpml = GPRADFWI.init_cpml(config)

        # Interior cells (away from PML) should have b=0, c=0
        mid_x = config.nx ÷ 2
        mid_y = config.ny ÷ 2
        @test cpml.be_x[mid_x] == 0.0  # no PML in interior
        @test cpml.ce_x[mid_x] == 0.0
        @test cpml.be_y[mid_y] == 0.0
        @test cpml.ce_y[mid_y] == 0.0
    end

    # ── Category B: Auxiliary field sizes ──────────────────────────────────

    @testset "Auxiliary ψ field dimensions" begin
        npml = 8
        config = create_config(
            nx=50, ny=40, dx=0.005, fc=500e6, npml=npml,
            src_ix=25, src_iy=10, rx_iy=5, rx_ix_list=[25],
            nt=10
        )
        cpml = GPRADFWI.init_cpml(config)

        # ψ fields should be sized (npml × ny) for x-boundaries
        @test size(cpml.psi_ezx_x1) == (npml, config.ny)
        @test size(cpml.psi_ezx_x2) == (npml, config.ny)
        @test size(cpml.psi_hyx_x1) == (npml, config.ny)
        @test size(cpml.psi_hyx_x2) == (npml, config.ny)

        # ψ fields should be sized (nx × npml) for y-boundaries
        @test size(cpml.psi_ezy_y1) == (config.nx, npml)
        @test size(cpml.psi_ezy_y2) == (config.nx, npml)
        @test size(cpml.psi_hxy_y1) == (config.nx, npml)
        @test size(cpml.psi_hxy_y2) == (config.nx, npml)

        # All initialized to zero
        @test all(cpml.psi_ezx_x1 .== 0.0)
        @test all(cpml.psi_ezy_y1 .== 0.0)
    end
end
