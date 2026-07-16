# hand_adjoint.jl -- Explicit reverse for the production Debye+CPML update.

"""
Store-all time-boundary states for the matched hand-adjoint forward solve.

Index `n + 1` contains the state after time step `n`; index 1 is the zero
initial state. CPML arrays retain the compact strip layout used by `CPMLData`.
"""
struct HandAdjointTape
    Ez::Array{Float64,3}
    Hx::Array{Float64,3}
    Hy::Array{Float64,3}
    Pz::Array{Float64,3}
    psi_ezx_x1::Array{Float64,3}
    psi_ezx_x2::Array{Float64,3}
    psi_hyx_x1::Array{Float64,3}
    psi_hyx_x2::Array{Float64,3}
    psi_ezy_y1::Array{Float64,3}
    psi_ezy_y2::Array{Float64,3}
    psi_hxy_y1::Array{Float64,3}
    psi_hxy_y2::Array{Float64,3}
end

function HandAdjointTape(config::FDTDConfig)
    nx, ny, nt = config.nx, config.ny, config.nt
    p = config.cpml.npml
    ns = nt + 1
    return HandAdjointTape(
        zeros(nx, ny, ns), zeros(nx, ny, ns),
        zeros(nx, ny, ns), zeros(nx, ny, ns),
        zeros(p, ny, ns), zeros(p, ny, ns),
        zeros(p, ny, ns), zeros(p, ny, ns),
        zeros(nx, p, ns), zeros(nx, p, ns),
        zeros(nx, p, ns), zeros(nx, p, ns),
    )
end

"""
    hand_adjoint_store_all_bytes(config)

Payload bytes used by the twelve Float64 state histories in `HandAdjointTape`.
This excludes Julia array headers, receiver data, coefficients, and reverse
scratch arrays.
"""
function hand_adjoint_store_all_bytes(config::FDTDConfig)
    nx, ny, nt = config.nx, config.ny, config.nt
    p = config.cpml.npml
    values_per_state = 4 * nx * ny + 4 * p * ny + 4 * nx * p
    return sizeof(Float64) * (nt + 1) * values_per_state
end

"""
    run_forward_hand_tape(config, eps_inf, deps, tau, sigma, src_waveform)

Run the same single-pole Debye TM update as the corrected production solver
and retain every time-boundary state needed by the explicit reverse. The
right/top H-side CPML strips start at `n - npml`, whereas right/top E-side
strips start at `n - npml + 1`, as required by Yee staggering.

The soft source updates both Ez and Debye polarization so that
`P_new = c1 * P_old + c2 * E_new` remains true at the source cell.
"""
function run_forward_hand_tape(config::FDTDConfig,
                               eps_inf::Matrix{Float64},
                               deps::Matrix{Float64},
                               tau::Matrix{Float64},
                               sigma::Matrix{Float64},
                               src_waveform::Vector{Float64})
    nx, ny, nt = config.nx, config.ny, config.nt
    dx, dy, dt = config.dx, config.dy, config.dt
    p = config.cpml.npml
    si, sj = config.source.ix, config.source.iy

    @assert size(eps_inf) == (nx, ny) "eps_inf size mismatch"
    @assert size(deps) == (nx, ny) "deps size mismatch"
    @assert size(tau) == (nx, ny) "tau size mismatch"
    @assert size(sigma) == (nx, ny) "sigma size mismatch"
    @assert length(src_waveform) == nt "source length mismatch"
    @assert length(config.rx_ix) == length(config.rx_iy) "receiver size mismatch"
    @assert 0 <= p < min(nx, ny) "invalid CPML thickness"
    @assert 2 <= si <= nx - 1 && 2 <= sj <= ny - 1 "source must be an updated E node"

    dc = init_debye_coeffs(eps_inf, deps, tau, sigma, dt, nx, ny)
    cpml = init_cpml(config)
    tape = HandAdjointTape(config)
    rec = zeros(nt, length(config.rx_ix))
    ax = dt / (mu0 * dx)
    ay = dt / (mu0 * dy)
    source_scale = 1.0 / (dx * dy)

    @inbounds for n in 1:nt
        n1 = n + 1

        # Carry the complete state. Updated entries below overwrite the copy.
        for j in 1:ny, i in 1:nx
            tape.Ez[i, j, n1] = tape.Ez[i, j, n]
            tape.Hx[i, j, n1] = tape.Hx[i, j, n]
            tape.Hy[i, j, n1] = tape.Hy[i, j, n]
            tape.Pz[i, j, n1] = tape.Pz[i, j, n]
        end
        for j in 1:ny, q in 1:p
            tape.psi_ezx_x1[q, j, n1] = tape.psi_ezx_x1[q, j, n]
            tape.psi_ezx_x2[q, j, n1] = tape.psi_ezx_x2[q, j, n]
            tape.psi_hyx_x1[q, j, n1] = tape.psi_hyx_x1[q, j, n]
            tape.psi_hyx_x2[q, j, n1] = tape.psi_hyx_x2[q, j, n]
        end
        for q in 1:p, i in 1:nx
            tape.psi_ezy_y1[i, q, n1] = tape.psi_ezy_y1[i, q, n]
            tape.psi_ezy_y2[i, q, n1] = tape.psi_ezy_y2[i, q, n]
            tape.psi_hxy_y1[i, q, n1] = tape.psi_hxy_y1[i, q, n]
            tape.psi_hxy_y2[i, q, n1] = tape.psi_hxy_y2[i, q, n]
        end

        # Hx: bottom and top CPML operate on y-directed E differences.
        for j in 1:ny-1, i in 1:nx
            dE = tape.Ez[i, j + 1, n] - tape.Ez[i, j, n]
            h = tape.Hx[i, j, n] - ay * dE / cpml.kappa_hy[j]
            if j <= p
                qnew = cpml.bh_y[j] * tape.psi_hxy_y1[i, j, n] + cpml.ch_y[j] * dE
                tape.psi_hxy_y1[i, j, n1] = qnew
                h -= ay * qnew
            end
            jj = j - (ny - p) + 1
            if 1 <= jj <= p
                qnew = cpml.bh_y[j] * tape.psi_hxy_y2[i, jj, n] + cpml.ch_y[j] * dE
                tape.psi_hxy_y2[i, jj, n1] = qnew
                h -= ay * qnew
            end
            tape.Hx[i, j, n1] = h
        end

        # Hy: left and right CPML operate on x-directed E differences.
        for j in 1:ny, i in 1:nx-1
            dE = tape.Ez[i + 1, j, n] - tape.Ez[i, j, n]
            h = tape.Hy[i, j, n] + ax * dE / cpml.kappa_hx[i]
            if i <= p
                qnew = cpml.bh_x[i] * tape.psi_hyx_x1[i, j, n] + cpml.ch_x[i] * dE
                tape.psi_hyx_x1[i, j, n1] = qnew
                h += ax * qnew
            end
            ii = i - (nx - p) + 1
            if 1 <= ii <= p
                qnew = cpml.bh_x[i] * tape.psi_hyx_x2[ii, j, n] + cpml.ch_x[i] * dE
                tape.psi_hyx_x2[ii, j, n1] = qnew
                h += ax * qnew
            end
            tape.Hy[i, j, n1] = h
        end

        # E-side CPML, combined E update, and Debye ADE.
        for j in 2:ny-1, i in 2:nx-1
            dHy = tape.Hy[i, j, n1] - tape.Hy[i - 1, j, n1]
            dHx = tape.Hx[i, j, n1] - tape.Hx[i, j - 1, n1]
            curl_h = dHy / (cpml.kappa_ex[i] * dx) -
                     dHx / (cpml.kappa_ey[j] * dy)
            curl_cpml = 0.0

            if i <= p
                qnew = cpml.be_x[i] * tape.psi_ezx_x1[i, j, n] + cpml.ce_x[i] * dHy
                tape.psi_ezx_x1[i, j, n1] = qnew
                curl_cpml += qnew / dx
            end
            ii = i - (nx - p)
            if 1 <= ii <= p
                qnew = cpml.be_x[i] * tape.psi_ezx_x2[ii, j, n] + cpml.ce_x[i] * dHy
                tape.psi_ezx_x2[ii, j, n1] = qnew
                curl_cpml += qnew / dx
            end
            if j <= p
                qnew = cpml.be_y[j] * tape.psi_ezy_y1[i, j, n] + cpml.ce_y[j] * dHx
                tape.psi_ezy_y1[i, j, n1] = qnew
                curl_cpml -= qnew / dy
            end
            jj = j - (ny - p)
            if 1 <= jj <= p
                qnew = cpml.be_y[j] * tape.psi_ezy_y2[i, jj, n] + cpml.ce_y[j] * dHx
                tape.psi_ezy_y2[i, jj, n1] = qnew
                curl_cpml -= qnew / dy
            end

            eold = tape.Ez[i, j, n]
            pold = tape.Pz[i, j, n]
            enew = dc.ca[i, j] * eold + dc.cb[i, j] * (curl_h + curl_cpml) +
                   dc.cp[i, j] * pold
            tape.Ez[i, j, n1] = enew
            tape.Pz[i, j, n1] = dc.c1[i, j] * pold + dc.c2[i, j] * enew
        end

        delta_e = dc.cb[si, sj] * src_waveform[n] * source_scale
        tape.Ez[si, sj, n1] += delta_e
        tape.Pz[si, sj, n1] += dc.c2[si, sj] * delta_e

        for r in eachindex(config.rx_ix)
            rec[n, r] = tape.Ez[config.rx_ix[r], config.rx_iy[r], n1]
        end
    end

    return rec, tape
end

"""
    hand_adjoint_gradient(config, obs_data, src_waveform, eps_inf,
                          deps, tau, sigma, param_mask; diagnostics=false)

Return the exact hand-derived gradient of `0.5 * ||synthetic-observed||^2`
with respect to `eps_inf` cells selected by `param_mask`. Flat gradients use
the package's column-major packing order: `j` outer, `i` inner.

Set `diagnostics=true` to also return the synthetic data, forward tape, full
gradient map, and store-all payload estimate.
"""
function hand_adjoint_gradient(config::FDTDConfig,
                               obs_data::Matrix{Float64},
                               src_waveform::Vector{Float64},
                               eps_inf::Matrix{Float64},
                               deps::Matrix{Float64},
                               tau::Matrix{Float64},
                               sigma::Matrix{Float64},
                               param_mask::AbstractMatrix{Bool};
                               diagnostics::Bool=false)
    nx, ny, nt = config.nx, config.ny, config.nt
    dx, dy, dt = config.dx, config.dy, config.dt
    p = config.cpml.npml
    si, sj = config.source.ix, config.source.iy
    nrx = length(config.rx_ix)

    @assert size(obs_data) == (nt, nrx) "observed-data size mismatch"
    @assert size(param_mask) == (nx, ny) "parameter-mask size mismatch"

    syn, tape = run_forward_hand_tape(config, eps_inf, deps, tau, sigma, src_waveform)
    dc = init_debye_coeffs(eps_inf, deps, tau, sigma, dt, nx, ny)
    cpml = init_cpml(config)
    ax = dt / (mu0 * dx)
    ay = dt / (mu0 * dy)
    source_scale = 1.0 / (dx * dy)
    grad_map = zeros(nx, ny)

    # Adjoint state at the output of the current reverse step.
    lE = zeros(nx, ny); lEold = similar(lE)
    lP = zeros(nx, ny); lPold = similar(lP)
    lHx = zeros(nx, ny); lHxold = similar(lHx)
    lHy = zeros(nx, ny); lHyold = similar(lHy)

    l_ex1 = zeros(p, ny); l_ex1_old = similar(l_ex1)
    l_ex2 = zeros(p, ny); l_ex2_old = similar(l_ex2)
    l_hyx1 = zeros(p, ny); l_hyx1_old = similar(l_hyx1)
    l_hyx2 = zeros(p, ny); l_hyx2_old = similar(l_hyx2)
    l_ey1 = zeros(nx, p); l_ey1_old = similar(l_ey1)
    l_ey2 = zeros(nx, p); l_ey2_old = similar(l_ey2)
    l_hxy1 = zeros(nx, p); l_hxy1_old = similar(l_hxy1)
    l_hxy2 = zeros(nx, p); l_hxy2_old = similar(l_hxy2)

    @inbounds for n in nt:-1:1
        n1 = n + 1

        for r in 1:nrx
            ri, rj = config.rx_ix[r], config.rx_iy[r]
            lE[ri, rj] += syn[n, r] - obs_data[n, r]
        end

        # Identity copies are correct for untouched boundary entries. Updated
        # entries are overwritten by their local reverse equations below.
        copyto!(lEold, lE); copyto!(lPold, lP)
        copyto!(l_ex1_old, l_ex1); copyto!(l_ex2_old, l_ex2)
        copyto!(l_hyx1_old, l_hyx1); copyto!(l_hyx2_old, l_hyx2)
        copyto!(l_ey1_old, l_ey1); copyto!(l_ey2_old, l_ey2)
        copyto!(l_hxy1_old, l_hxy1); copyto!(l_hxy2_old, l_hxy2)

        # Reverse the E-side updates in exact opposite grid order.
        for j in ny-1:-1:2, i in nx-1:-1:2
            dHy = tape.Hy[i, j, n1] - tape.Hy[i - 1, j, n1]
            dHx = tape.Hx[i, j, n1] - tape.Hx[i, j - 1, n1]
            curl_h = dHy / (cpml.kappa_ex[i] * dx) -
                     dHx / (cpml.kappa_ey[j] * dy)
            curl_cpml = 0.0
            if i <= p
                curl_cpml += tape.psi_ezx_x1[i, j, n1] / dx
            end
            ii = i - (nx - p)
            if 1 <= ii <= p
                curl_cpml += tape.psi_ezx_x2[ii, j, n1] / dx
            end
            if j <= p
                curl_cpml -= tape.psi_ezy_y1[i, j, n1] / dy
            end
            jj = j - (ny - p)
            if 1 <= jj <= p
                curl_cpml -= tape.psi_ezy_y2[i, jj, n1] / dy
            end

            source_term = (i == si && j == sj) ? src_waveform[n] * source_scale : 0.0
            forcing = curl_h + curl_cpml + source_term
            lp = lP[i, j]
            le = lE[i, j] + dc.c2[i, j] * lp

            lEold[i, j] = dc.ca[i, j] * le
            lPold[i, j] = dc.c1[i, j] * lp + dc.cp[i, j] * le

            # c1 and c2 are independent of eps_inf when deps/tau are fixed.
            denom = eps0 * eps_inf[i, j] + dc.c2[i, j] + sigma[i, j] * dt / 2.0
            numer = eps0 * eps_inf[i, j] - sigma[i, j] * dt / 2.0
            inv_denom2 = 1.0 / (denom * denom)
            dca = eps0 * (denom - numer) * inv_denom2
            dcb = -eps0 * dt * inv_denom2
            dcp = (tau[i, j] > 0.0 && deps[i, j] != 0.0) ?
                   -eps0 * (1.0 - dc.c1[i, j]) * inv_denom2 : 0.0
            grad_map[i, j] += le * (dca * tape.Ez[i, j, n] +
                                     dcb * forcing + dcp * tape.Pz[i, j, n])

            lc = dc.cb[i, j] * le
            ldHy = lc / (cpml.kappa_ex[i] * dx)
            ldHx = -lc / (cpml.kappa_ey[j] * dy)

            # Reverse forward order: y-top, y-bottom, x-right, x-left.
            if 1 <= jj <= p
                lq = l_ey2[i, jj] - lc / dy
                l_ey2_old[i, jj] = cpml.be_y[j] * lq
                ldHx += cpml.ce_y[j] * lq
            end
            if j <= p
                lq = l_ey1[i, j] - lc / dy
                l_ey1_old[i, j] = cpml.be_y[j] * lq
                ldHx += cpml.ce_y[j] * lq
            end
            if 1 <= ii <= p
                lq = l_ex2[ii, j] + lc / dx
                l_ex2_old[ii, j] = cpml.be_x[i] * lq
                ldHy += cpml.ce_x[i] * lq
            end
            if i <= p
                lq = l_ex1[i, j] + lc / dx
                l_ex1_old[i, j] = cpml.be_x[i] * lq
                ldHy += cpml.ce_x[i] * lq
            end

            lHy[i, j] += ldHy
            lHy[i - 1, j] -= ldHy
            lHx[i, j] += ldHx
            lHx[i, j - 1] -= ldHx
        end

        # Reverse E has now added the current-step curl contribution to lH.
        # Copy the complete output adjoint through H_new = H_old + update(E).
        copyto!(lHxold, lHx)
        copyto!(lHyold, lHy)

        # Reverse Hy, including right/left CPML recurrences.
        for j in ny:-1:1, i in nx-1:-1:1
            lam = lHy[i, j]
            ldE = ax * lam / cpml.kappa_hx[i]
            ii = i - (nx - p) + 1
            if 1 <= ii <= p
                lq = l_hyx2[ii, j] + ax * lam
                l_hyx2_old[ii, j] = cpml.bh_x[i] * lq
                ldE += cpml.ch_x[i] * lq
            end
            if i <= p
                lq = l_hyx1[i, j] + ax * lam
                l_hyx1_old[i, j] = cpml.bh_x[i] * lq
                ldE += cpml.ch_x[i] * lq
            end
            lEold[i + 1, j] += ldE
            lEold[i, j] -= ldE
        end

        # Reverse Hx, including top/bottom CPML recurrences.
        for j in ny-1:-1:1, i in nx:-1:1
            lam = lHx[i, j]
            ldE = -ay * lam / cpml.kappa_hy[j]
            jj = j - (ny - p) + 1
            if 1 <= jj <= p
                lq = l_hxy2[i, jj] - ay * lam
                l_hxy2_old[i, jj] = cpml.bh_y[j] * lq
                ldE += cpml.ch_y[j] * lq
            end
            if j <= p
                lq = l_hxy1[i, j] - ay * lam
                l_hxy1_old[i, j] = cpml.bh_y[j] * lq
                ldE += cpml.ch_y[j] * lq
            end
            lEold[i, j + 1] += ldE
            lEold[i, j] -= ldE
        end

        tmp = lE; lE = lEold; lEold = tmp
        tmp = lP; lP = lPold; lPold = tmp
        tmp = lHx; lHx = lHxold; lHxold = tmp
        tmp = lHy; lHy = lHyold; lHyold = tmp
        tmp = l_ex1; l_ex1 = l_ex1_old; l_ex1_old = tmp
        tmp = l_ex2; l_ex2 = l_ex2_old; l_ex2_old = tmp
        tmp = l_hyx1; l_hyx1 = l_hyx1_old; l_hyx1_old = tmp
        tmp = l_hyx2; l_hyx2 = l_hyx2_old; l_hyx2_old = tmp
        tmp = l_ey1; l_ey1 = l_ey1_old; l_ey1_old = tmp
        tmp = l_ey2; l_ey2 = l_ey2_old; l_ey2_old = tmp
        tmp = l_hxy1; l_hxy1 = l_hxy1_old; l_hxy1_old = tmp
        tmp = l_hxy2; l_hxy2 = l_hxy2_old; l_hxy2_old = tmp
    end

    grad = Vector{Float64}(undef, count(param_mask))
    k = 0
    @inbounds for j in 1:ny, i in 1:nx
        if param_mask[i, j]
            k += 1
            grad[k] = grad_map[i, j]
        end
    end

    if diagnostics
        return (
            gradient=grad,
            gradient_map=grad_map,
            synthetic=syn,
            tape=tape,
            store_all_bytes=hand_adjoint_store_all_bytes(config),
        )
    end
    return grad
end

"""
Flat-parameter convenience overload matching `forward_misfit` packing.
"""
function hand_adjoint_gradient(params_flat::Vector{Float64},
                               config::FDTDConfig,
                               obs_data::Matrix{Float64},
                               src_waveform::Vector{Float64},
                               eps_inf_bg::Matrix{Float64},
                               deps::Matrix{Float64},
                               tau::Matrix{Float64},
                               sigma::Matrix{Float64},
                               param_mask::AbstractMatrix{Bool};
                               diagnostics::Bool=false)
    @assert length(params_flat) == count(param_mask) "parameter count mismatch"
    eps_inf = copy(eps_inf_bg)
    k = 0
    @inbounds for j in axes(param_mask, 2), i in axes(param_mask, 1)
        if param_mask[i, j]
            k += 1
            eps_inf[i, j] = params_flat[k]
        end
    end
    return hand_adjoint_gradient(
        config, obs_data, src_waveform, eps_inf, deps, tau, sigma, param_mask;
        diagnostics=diagnostics,
    )
end
