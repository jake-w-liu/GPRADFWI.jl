# sources.jl — Source waveforms for GPR simulation
#
# Time convention: e^{+iωt}

"""
    ricker_wavelet(t, fc, t0)

Ricker wavelet (second derivative of Gaussian) centered at time `t0`
with center frequency `fc`.

    s(t) = (1 - 2π²fc²(t-t0)²) exp(-π²fc²(t-t0)²)
"""
function ricker_wavelet(t::Float64, fc::Float64, t0::Float64)
    tau = t - t0
    arg = (π * fc * tau)^2
    return (1.0 - 2.0 * arg) * exp(-arg)
end

"""
    create_source(config::FDTDConfig)

Pre-compute the source time series for all time steps.
Returns a vector of length config.nt.
"""
function create_source(config::FDTDConfig)
    src = Vector{Float64}(undef, config.nt)
    for n in 1:config.nt
        t = (n - 0.5) * config.dt  # half-step offset for Ez update
        src[n] = ricker_wavelet(t, config.source.fc, config.source.t0)
    end
    return src
end
