module GPRADFWI

using LinearAlgebra
using Printf
using Random
using DelimitedFiles
using Statistics

# ── Constants ──────────────────────────────────────────────────────────
"""
    c0

Speed of light in vacuum in meters per second.
"""
const c0   = 299792458.0        # speed of light [m/s]

"""
    eps0

Vacuum permittivity in farads per meter.
"""
const eps0 = 8.854187817e-12    # vacuum permittivity [F/m]

"""
    mu0

Vacuum permeability in henries per meter.
"""
const mu0  = 4π * 1e-7          # vacuum permeability [H/m]

"""
    eta0

Free-space impedance in ohms.
"""
const eta0 = sqrt(mu0 / eps0)   # free-space impedance [Ω]

export c0, eps0, mu0, eta0

# ── Includes ───────────────────────────────────────────────────────────
include("types.jl")
include("sources.jl")
include("cpml.jl")
include("debye.jl")
include("fdtd.jl")
include("inversion.jl")

# ── Exports ────────────────────────────────────────────────────────────
export FDTDConfig, DebyeMedium, CPMLParams, SourceConfig
export FWIResult
export create_config, create_source
export run_forward!, run_forward_snapshots, compute_misfit, forward_misfit
export fd_gradient, ad_gradient
export run_fwi, run_fwi_multisource

end # module
