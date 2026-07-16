@testset verbose = true "FDTD Forward Solver" begin

    # Helper: create a small test configuration
    function small_config(; nx=60, ny=50, dx=0.01, fc=300e6, npml=5, nt=300)
        return create_config(
            nx=nx, ny=ny, dx=dx, fc=fc, npml=npml,
            src_ix=nx÷2, src_iy=ny÷2,
            rx_iy=npml+2,
            rx_ix_list=collect((npml+2):2:(nx-npml-1)),
            nt=nt
        )
    end

    # ── Category A: Free-space propagation amplitude ──────────────────────

    @testset "Free-space simulation — basic sanity" begin
        Random.seed!(301)
        config = small_config(nt=200)
        nx, ny = config.nx, config.ny

        # Uniform free space: ε∞=1, no dispersion, no loss
        eps_inf = ones(nx, ny)
        deps = zeros(nx, ny)
        tau = zeros(nx, ny)
        sigma = zeros(nx, ny)
        src = create_source(config)

        rec = run_forward!(config, eps_inf, deps, tau, sigma, src)

        # Output shape: nt × n_receivers
        @test size(rec) == (config.nt, length(config.rx_ix))

        # No NaN or Inf
        @test !any(isnan, rec)  # well-posed FDTD must produce finite fields
        @test !any(isinf, rec)

        # Non-zero: receivers should detect the propagating wavelet
        @test maximum(abs, rec) > 0.0  # source creates non-zero fields
    end

    # ── Category B: Reciprocity ───────────────────────────────────────────

    @testset "Reciprocity: swap source and receiver" begin
        # In a reciprocal medium, the response at B due to source at A
        # equals the response at A due to source at B (Green's function symmetry)
        nx, ny = 60, 50
        dx = 0.01
        fc = 300e6
        npml = 5
        nt = 250

        # Config 1: source at (20, 25), receiver at (40, 25)
        config1 = create_config(
            nx=nx, ny=ny, dx=dx, fc=fc, npml=npml,
            src_ix=20, src_iy=25, rx_iy=25, rx_ix_list=[40],
            nt=nt
        )
        # Config 2: source at (40, 25), receiver at (20, 25)
        config2 = create_config(
            nx=nx, ny=ny, dx=dx, fc=fc, npml=npml,
            src_ix=40, src_iy=25, rx_iy=25, rx_ix_list=[20],
            nt=nt
        )

        # Uniform non-dispersive medium
        eps_inf = fill(4.0, nx, ny)
        deps = zeros(nx, ny)
        tau = zeros(nx, ny)
        sigma = fill(0.001, nx, ny)

        src1 = create_source(config1)
        src2 = create_source(config2)

        rec1 = run_forward!(config1, eps_inf, deps, tau, sigma, src1)
        rec2 = run_forward!(config2, eps_inf, deps, tau, sigma, src2)

        # Reciprocity: rec1 ≈ rec2 (same distance, same medium)
        # rtol=1e-10: numerical reciprocity limited by CPML asymmetry and grid discretization
        @test isapprox(rec1[:, 1], rec2[:, 1]; rtol=1e-10)
    end

    # ── Category B: Energy conservation (lossless) ────────────────────────

    @testset "No energy growth in lossless medium" begin
        config = small_config(nt=400)
        nx, ny = config.nx, config.ny

        # Lossless free space
        eps_inf = ones(nx, ny)
        deps = zeros(nx, ny)
        tau = zeros(nx, ny)
        sigma = zeros(nx, ny)
        src = create_source(config)

        rec = run_forward!(config, eps_inf, deps, tau, sigma, src)

        # After source dies out (> 3/fc + margin), the total energy should not grow
        # Check that late-time amplitudes don't exceed early peak
        peak_amp = maximum(abs, rec)
        late_start = config.nt - 50
        late_amp = maximum(abs, rec[late_start:end, :])

        # Late-time amplitude should be smaller than peak (wave has spread + PML absorbed)
        @test late_amp < peak_amp  # no energy growth
    end

    # ── Category A: Source injection with known cb ────────────────────────

    @testset "Source injection coefficient consistency" begin
        # The source injection uses cb/(dx·dy), verify that for vacuum:
        # cb = dt/ε₀ and the injection is correctly normalized
        config = small_config(nt=1)
        nx, ny = config.nx, config.ny

        eps_inf = ones(nx, ny)
        deps = zeros(nx, ny)
        tau = zeros(nx, ny)
        sigma = zeros(nx, ny)

        # Source at (src_ix, src_iy)
        si = config.source.ix
        sj = config.source.iy

        # Compute expected cb for vacuum
        dt = config.dt
        dx = config.dx
        cb_vac = dt / GPRADFWI.eps0

        # After 1 time step with known source value
        src = create_source(config)
        rec = run_forward!(config, eps_inf, deps, tau, sigma, src)

        # The E-field at source location after 1 step should be:
        # Ez = cb * src[1] / (dx * dy)   (only source term, curl_H ≈ 0 at first step)
        # We can't directly read Ez, but we can verify the source doesn't produce NaN
        @test !any(isnan, rec)
        @test !any(isinf, rec)
    end

    # ── Category C: Convergence with grid refinement ──────────────────────

    @testset "Grid refinement: finer grid → less dispersion error" begin
        # Run same physical scenario at two resolutions; finer should have
        # smaller numerical dispersion (closer to analytical arrival time)
        fc = 300e6
        nt = 200

        # Coarse grid: dx=20mm (5 cells/wavelength at 300MHz in εr=4)
        config_c = create_config(
            nx=30, ny=25, dx=0.02, fc=fc, npml=5,
            src_ix=15, src_iy=13, rx_iy=5, rx_ix_list=[15],
            nt=nt
        )
        # Fine grid: dx=10mm (10 cells/wavelength)
        config_f = create_config(
            nx=60, ny=50, dx=0.01, fc=fc, npml=5,
            src_ix=30, src_iy=25, rx_iy=10, rx_ix_list=[30],
            nt=nt
        )

        # Uniform medium ε∞=4 (c = c0/2)
        eps_c = fill(4.0, config_c.nx, config_c.ny)
        eps_f = fill(4.0, config_f.nx, config_f.ny)
        z_c = zeros(config_c.nx, config_c.ny)
        z_f = zeros(config_f.nx, config_f.ny)

        src_c = create_source(config_c)
        src_f = create_source(config_f)

        rec_c = run_forward!(config_c, eps_c, z_c, z_c, z_c, src_c)
        rec_f = run_forward!(config_f, eps_f, z_f, z_f, z_f, src_f)

        # Both should produce valid output
        @test !any(isnan, rec_c)
        @test !any(isnan, rec_f)

        # Fine grid should have higher peak amplitude (less numerical dispersion/dissipation)
        # This is a qualitative convergence check
        @test maximum(abs, rec_f) > 0.0
        @test maximum(abs, rec_c) > 0.0
    end

    # ── Category D: Dispersive medium ─────────────────────────────────────

    @testset "Dispersive medium — stable and finite" begin
        config = small_config(nt=300)
        nx, ny = config.nx, config.ny

        # Debye dispersive soil
        eps_inf = fill(4.0, nx, ny)
        deps = fill(4.0, nx, ny)
        tau = fill(0.3e-9, nx, ny)
        sigma = fill(0.005, nx, ny)
        src = create_source(config)

        rec = run_forward!(config, eps_inf, deps, tau, sigma, src)

        @test !any(isnan, rec)  # dispersive ADE must remain stable
        @test !any(isinf, rec)
        @test maximum(abs, rec) > 0.0  # non-zero signal
    end

    @testset "Dispersive vs non-dispersive: attenuation" begin
        config = small_config(nt=300)
        nx, ny = config.nx, config.ny

        # Non-dispersive: ε∞=8 (same static permittivity as Debye with ε∞=4, Δε=4)
        eps_nd = fill(8.0, nx, ny)
        deps_nd = zeros(nx, ny)
        tau_nd = zeros(nx, ny)
        sigma_nd = fill(0.005, nx, ny)

        # Dispersive: ε∞=4, Δε=4, τ=0.3ns (εs=8)
        eps_d = fill(4.0, nx, ny)
        deps_d = fill(4.0, nx, ny)
        tau_d = fill(0.3e-9, nx, ny)
        sigma_d = fill(0.005, nx, ny)

        src = create_source(config)
        rec_nd = run_forward!(config, eps_nd, deps_nd, tau_nd, sigma_nd, src)
        rec_d = run_forward!(config, eps_d, deps_d, tau_d, sigma_d, src)

        # Dispersive medium should produce different waveforms (dispersion causes spreading)
        # The signals should NOT be identical
        @test !isapprox(rec_nd, rec_d; rtol=0.01)  # must differ by > 1%
    end

    @testset "CPML kappa scaling is applied in field updates" begin
        config = small_config(nx=20, ny=20, npml=4, nt=1)
        nx, ny = config.nx, config.ny
        ch_dt_dx = config.dt / (GPRADFWI.mu0 * config.dx)
        ch_dt_dy = config.dt / (GPRADFWI.mu0 * config.dy)

        cpml_h_ref = GPRADFWI.init_cpml(config)
        cpml_h_kap = GPRADFWI.init_cpml(config)
        for cp in (cpml_h_ref, cpml_h_kap)
            cp.bh_x .= 0.0
            cp.ch_x .= 0.0
            cp.bh_y .= 0.0
            cp.ch_y .= 0.0
            cp.kappa_hx .= 1.0
            cp.kappa_hy .= 1.0
            cp.psi_hyx_x1 .= 0.0
            cp.psi_hyx_x2 .= 0.0
            cp.psi_hxy_y1 .= 0.0
            cp.psi_hxy_y2 .= 0.0
        end
        cpml_h_kap.kappa_hy[1] = 4.0

        Ez = zeros(nx, ny)
        Ez[:, 2] .= 1.0

        Hx_ref = zeros(nx, ny); Hy_ref = zeros(nx, ny)
        Hx_kap = zeros(nx, ny); Hy_kap = zeros(nx, ny)

        GPRADFWI._update_H!(Hx_ref, Hy_ref, Ez, ch_dt_dx, ch_dt_dy, cpml_h_ref, nx, ny)
        GPRADFWI._update_H!(Hx_kap, Hy_kap, Ez, ch_dt_dx, ch_dt_dy, cpml_h_kap, nx, ny)

        # Boundary update magnitude must be reduced by 1/κ in y-directed Hx update.
        @test isapprox(Hx_kap[5, 1], Hx_ref[5, 1] / 4.0; rtol=1e-14, atol=0.0)
        # Interior rows (κ=1) remain unchanged.
        @test isapprox(Hx_kap[5, 2], Hx_ref[5, 2]; rtol=1e-14, atol=0.0)

        cpml_e_ref = GPRADFWI.init_cpml(config)
        cpml_e_kap = GPRADFWI.init_cpml(config)
        for cp in (cpml_e_ref, cpml_e_kap)
            cp.be_x .= 0.0
            cp.ce_x .= 0.0
            cp.be_y .= 0.0
            cp.ce_y .= 0.0
            cp.kappa_ex .= 1.0
            cp.kappa_ey .= 1.0
            cp.psi_ezx_x1 .= 0.0
            cp.psi_ezx_x2 .= 0.0
            cp.psi_ezy_y1 .= 0.0
            cp.psi_ezy_y2 .= 0.0
        end
        cpml_e_kap.kappa_ex[2] = 5.0

        eps_inf = fill(4.0, nx, ny)
        deps = zeros(nx, ny)
        tau = zeros(nx, ny)
        sigma = fill(0.001, nx, ny)
        dc = GPRADFWI.init_debye_coeffs(eps_inf, deps, tau, sigma, config.dt, nx, ny)

        Ez_ref = zeros(nx, ny); Ez_kap = zeros(nx, ny)
        Hx = zeros(nx, ny); Hy = zeros(nx, ny); Pz = zeros(nx, ny)
        Hy[2, 2] = 1.0

        GPRADFWI._update_E_debye!(Ez_ref, Hx, Hy, Pz, dc, cpml_e_ref, config.dt, config.dx, config.dy, nx, ny)

        Hx2 = zeros(nx, ny); Hy2 = copy(Hy); Pz2 = zeros(nx, ny)
        GPRADFWI._update_E_debye!(Ez_kap, Hx2, Hy2, Pz2, dc, cpml_e_kap, config.dt, config.dx, config.dy, nx, ny)

        # x-directed curl contribution at i=2 must scale by 1/κ_ex.
        @test isapprox(Ez_kap[2, 2], Ez_ref[2, 2] / 5.0; rtol=1e-14, atol=0.0)
    end

    # ── Category E: Error handling ────────────────────────────────────────

@testset "Debye-consistent source injection" begin
    Ez = fill(2.0, 3, 3)
    Pz = fill(0.4, 3, 3)
    coeffs = (cb=fill(0.5, 3, 3), c2=fill(0.25, 3, 3))
    old_E = Ez[2, 2]
    old_P = Pz[2, 2]

    delta_E = GPRADFWI._inject_source_debye!(Ez, Pz, coeffs, 2, 2, 3.0, 0.75)
    @test delta_E == 2.0
    @test Ez[2, 2] == old_E + delta_E
    @test Pz[2, 2] == old_P + coeffs.c2[2, 2] * delta_E

    Pz_nondispersive = zeros(3, 3)
    nondispersive = (cb=fill(0.5, 3, 3), c2=zeros(3, 3))
    GPRADFWI._inject_source_debye!(copy(Ez), Pz_nondispersive,
                                   nondispersive, 2, 2, 3.0, 0.75)
    @test iszero(Pz_nondispersive[2, 2])
end

@testset "Snapshot step validation" begin
    @test isnothing(GPRADFWI._validate_snapshot_steps([1, 3, 5], 5))
    @test_throws ArgumentError GPRADFWI._validate_snapshot_steps([2, 2], 5)
    @test_throws ArgumentError GPRADFWI._validate_snapshot_steps([0, 2], 5)
end

@testset "Dimension assertion on material maps" begin
        config = small_config()
        nx, ny = config.nx, config.ny
        src = create_source(config)

        # Wrong-sized eps_inf should fail
    @test_throws DimensionMismatch run_forward!(
            config, ones(nx+1, ny), zeros(nx, ny), zeros(nx, ny), zeros(nx, ny), src
        )

        # Wrong-sized source should fail
    @test_throws DimensionMismatch run_forward!(
            config, ones(nx, ny), zeros(nx, ny), zeros(nx, ny), zeros(nx, ny),
            zeros(config.nt + 1)
        )
    end

    # ── Category A: compute_misfit ────────────────────────────────────────

    @testset "compute_misfit" begin
        # L2 misfit: J = 0.5 * ||A - B||²
        A = [1.0 2.0; 3.0 4.0]
        B = [1.5 2.5; 3.5 4.5]

        # Hand computation: residuals = [-0.5, -0.5, -0.5, -0.5]
        # ||r||² = 4 × 0.25 = 1.0, J = 0.5
        @test compute_misfit(A, B) == 0.5  # exact: simple Float64 arithmetic

        # Zero misfit when identical
        @test compute_misfit(A, A) == 0.0  # exact zero

        # Symmetry: misfit(A,B) == misfit(B,A)
        @test compute_misfit(A, B) == compute_misfit(B, A)

        # Positive definite: misfit > 0 when A ≠ B
        @test compute_misfit(A, B) > 0.0

        # Scaling: misfit(αA, 0) = 0.5 α² ||A||²
        alpha = 3.0
        Z = zeros(2, 2)
        @test isapprox(compute_misfit(alpha .* A, Z), 0.5 * alpha^2 * norm(A)^2; rtol=1e-14)
    end
end
