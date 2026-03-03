@testset verbose = true "Sources (Ricker Wavelet)" begin

    # Access internal function
    ricker = GPRADFWI.ricker_wavelet

    # ── Category A: Analytical ground truth ───────────────────────────────

    @testset "Peak value and location" begin
        fc = 500e6  # 500 MHz
        t0 = 3.0e-9  # 3 ns

        # At t = t0: τ=0, arg=0, s(t0) = (1 - 0)*exp(0) = 1.0 exactly
        @test ricker(t0, fc, t0) == 1.0  # exact: no floating-point error at τ=0

        # Verify peak is at t0 by sampling nearby points
        dt_small = 1e-12  # 1 ps
        val_at_peak = ricker(t0, fc, t0)
        val_before = ricker(t0 - dt_small, fc, t0)
        val_after = ricker(t0 + dt_small, fc, t0)
        # Peak must be a local maximum
        @test val_at_peak > val_before  # peak is higher than left neighbor
        @test val_at_peak > val_after   # peak is higher than right neighbor
    end

    @testset "Known values at specific times" begin
        fc = 1.0  # unit frequency for easy hand calculation
        t0 = 0.0  # zero delay

        # At t=0: s(0) = (1 - 0)*exp(0) = 1.0
        @test ricker(0.0, 1.0, 0.0) == 1.0

        # At t = 1/(π·fc) = 1/π: arg = (π·1·1/π)² = 1
        # s = (1 - 2·1)·exp(-1) = -exp(-1) ≈ -0.367879441...
        t_test = 1.0 / π
        expected = -exp(-1.0)
        @test isapprox(ricker(t_test, 1.0, 0.0), expected; atol=1e-15)  # atol=1e-15: direct Float64

        # At t = √(3)/(π·fc·√2): arg = (π·√3/(π√2))² = 3/2
        # s = (1 - 2·3/2)·exp(-3/2) = -2·exp(-3/2)
        t_test2 = sqrt(3.0) / (π * sqrt(2.0))
        expected2 = -2.0 * exp(-1.5)
        @test isapprox(ricker(t_test2, 1.0, 0.0), expected2; atol=1e-14)  # atol=1e-14: sqrt composition
    end

    @testset "Zero crossings" begin
        # s(t) = 0 when 1 - 2π²fc²τ² = 0, i.e., τ = ±1/(πfc√2)
        fc = 500e6
        t0 = 3.0e-9
        tau_zero = 1.0 / (π * fc * sqrt(2.0))  # ~4.50e-10 s

        val_plus = ricker(t0 + tau_zero, fc, t0)
        val_minus = ricker(t0 - tau_zero, fc, t0)
        @test isapprox(val_plus, 0.0; atol=1e-15)   # atol=1e-15: direct Float64 at exact zero
        @test isapprox(val_minus, 0.0; atol=1e-15)
    end

    # ── Category B: Mathematical properties ───────────────────────────────

    @testset "Symmetry about t0" begin
        Random.seed!(101)
        fc = 500e6
        t0 = 3.0e-9
        # Ricker wavelet is symmetric: s(t0+τ) = s(t0-τ)
        # Not bitwise exact due to (t0+τ) vs (t0-τ) rounding differently in Float64
        for _ in 1:5
            tau = rand() * 5e-9  # random offset within 5 ns
            # rtol=1e-13: Float64 round-off from different addition/subtraction paths
            @test isapprox(ricker(t0 + tau, fc, t0), ricker(t0 - tau, fc, t0); rtol=1e-13)
        end
    end

    @testset "Frequency scaling" begin
        # s(t; fc, t0) depends only on (fc·(t-t0)): scaling fc by α and τ by 1/α gives same value
        # Specifically: ricker(t0 + τ, fc, t0) = ricker(t0 + τ/α, α·fc, t0) when both use arg=(π·fc·τ)²
        fc = 500e6
        t0 = 0.0
        tau = 1e-9
        alpha = 2.0
        val1 = ricker(t0 + tau, fc, t0)
        val2 = ricker(t0 + tau / alpha, alpha * fc, t0)
        @test isapprox(val1, val2; atol=1e-15)  # atol=1e-15: same arithmetic path
    end

    @testset "Bounded amplitude" begin
        # |s(t)| ≤ 1 for all t (peak is exactly 1 at t=t0)
        Random.seed!(102)
        fc = 500e6
        t0 = 3.0e-9
        for _ in 1:20
            t = rand() * 20e-9  # sample over full simulation window
            @test abs(ricker(t, fc, t0)) <= 1.0 + eps()  # wavelet amplitude bounded by 1
        end
    end

    @testset "Exponential decay far from peak" begin
        # For large |t-t0|, the wavelet decays to ~0
        fc = 500e6
        t0 = 3.0e-9
        # At 10 periods away: arg = (π·fc·10/fc)² = (10π)² ≈ 987, exp(-987) ≈ 0
        t_far = t0 + 10.0 / fc
        @test abs(ricker(t_far, fc, t0)) < 1e-100  # effectively zero
    end

    # ── Category D: Edge cases ────────────────────────────────────────────

    @testset "Edge cases" begin
        # Zero delay: peak at t=0
        @test ricker(0.0, 500e6, 0.0) == 1.0

        # Very high frequency: should still compute without overflow
        val = ricker(0.0, 1e12, 0.0)
        @test val == 1.0  # at peak, always 1
    end

    # ── Category A: create_source waveform ────────────────────────────────

    @testset "create_source" begin
        config = create_config(
            nx=60, ny=50, dx=0.01, fc=300e6, npml=5,
            src_ix=30, src_iy=10, rx_iy=5, rx_ix_list=collect(10:50),
            nt=100
        )
        src = create_source(config)

        # Correct length
        @test length(src) == config.nt  # must match time steps

        # No NaN or Inf
        @test !any(isnan, src)  # Ricker wavelet is always finite
        @test !any(isinf, src)

        # Peak should be near t0 = 1.5/fc
        t0 = config.source.t0
        # The time sample closest to t0
        n_peak = round(Int, t0 / config.dt + 0.5)  # account for half-step offset
        if 1 <= n_peak <= config.nt
            # Peak sample should be the maximum (or close to it)
            peak_region = max(1, n_peak-2):min(config.nt, n_peak+2)
            @test argmax(src) ∈ peak_region  # peak within ±2 samples of expected
        end

        # Source should decay to near-zero at boundaries (if nt is large enough)
        # First sample at t = 0.5*dt << t0 = 1.5/fc should be small
        @test abs(src[1]) < 0.01  # first sample well before wavelet peak
    end
end
