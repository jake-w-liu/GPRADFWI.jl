@testset verbose = true "Physical Constants" begin
    # Category A: Analytical ground truth — verify against NIST/CODATA values

    # Speed of light: exact by definition (SI 2019)
    @test c0 == 299792458.0  # exact integer value [m/s]

    # Vacuum permittivity: ε₀ = 1/(μ₀c₀²)
    # CODATA 2018: 8.8541878128e-12 F/m (exact under SI 2019)
    @test isapprox(eps0, 8.8541878128e-12; rtol=1e-8)  # rtol=1e-8: code uses 8.854187817e-12 (10 sig figs)

    # Vacuum permeability: μ₀ = 4π×10⁻⁷ H/m (exact by definition pre-2019)
    @test isapprox(mu0, 4π * 1e-7; rtol=1e-15)  # rtol=1e-15: Float64 arithmetic only

    # Free-space impedance: η₀ = √(μ₀/ε₀) ≈ 376.730 Ω
    eta0_check = sqrt(mu0 / eps0)
    @test isapprox(eta0, eta0_check; rtol=1e-14)  # rtol=1e-14: single sqrt, well-conditioned

    # Category B: Mathematical property — self-consistency
    # c₀ = 1/√(μ₀ε₀) must hold
    c0_from_mu_eps = 1.0 / sqrt(mu0 * eps0)
    @test isapprox(c0, c0_from_mu_eps; rtol=1e-8)  # rtol=1e-8: limited by eps0 precision
end
