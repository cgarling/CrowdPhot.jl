# Simultaneous PSF fitting photometry for a single image.
#
# A whole-image fitting algorithm that optimizes the parameters of
# every source at once, swappable for the DOLPHOT-style sequential
# `fit_all_stars` with every other pipeline stage (background, detection,
# morphology, PSF model) held fixed.
#
# The solve object is the normal-equation matrix `H = J'WJ` (never `J`),
# accumulated directly from per-star stamp derivatives.  The default solver
# is preconditioned conjugate gradient on `H`; a CHOLMOD Cholesky solve is
# available as an exact oracle.

# ==============================================================================
# Stamp derivative operator
# ==============================================================================

"""
    StampDerivatives{T, I <: Integer}

Per-star Jacobian values, stored as stamps; `J` is never materialized.

# Fields

- `values`: `(p, S², n_active)` array of weighted, column-equilibrated
  derivatives (`raw ./ colnorm`), where `p` is the number of free parameters
  per star and `S²` the number of stamp pixels.
- `pixels`: `(S², n_active)` flat pixel indices into the image, with `0`
  marking masked or off-image pixels.
- `colnorm`: `(p, n_active)` true per-column norm, kept from fill time.
- `npix`: number of pixels in the (flattened) image.
- `p`: number of free parameters per star.
- `S2`: number of stamp pixels (`S²`).
"""
struct StampDerivatives{T, I <: Integer}
    values::Array{T, 3}
    pixels::Matrix{I}
    colnorm::Matrix{T}
    npix::Int
    p::Int
    S2::Int
end

"""
    apply_JT!(z, Jm, u)

Compute `z = Jm' * u` where `u` is the *weighted* residual
`sqrt.(w) .* (model .- data)` and `z` is the equilibrated gradient
`D⁻¹ J' r` (i.e. `b_scaled`).  `z` is filled in place.
"""
function apply_JT!(z::AbstractVector, Jm::StampDerivatives, u::AbstractVector, live)
    p = Jm.p
    S2 = Jm.S2
    n_active = size(Jm.values, 3)
    fill!(z, zero(eltype(z)))
    @inbounds for a in 0:(n_active - 1)
        live[a + 1] || continue
        base = a * p
        for k in 1:p
            acc = zero(eltype(z))
            for m in 1:S2
                fi = Jm.pixels[m, a + 1]
                fi != 0 || continue
                acc += Jm.values[k, m, a + 1] * u[fi]
            end
            z[base + k] = acc
        end
    end
    return z
end

"""
    apply_J!(y, Jm, v)

Compute `y = Jm * v` (additive, `y` filled in place).  Reference path only
(the default solver iterates on `H`, never `J`); used for the adjoint test.
"""
function apply_J!(y::AbstractVector, Jm::StampDerivatives, v::AbstractVector)
    p = Jm.p
    S2 = Jm.S2
    n_active = size(Jm.values, 3)
    fill!(y, zero(eltype(y)))
    @inbounds for a in 0:(n_active - 1)
        base = a * p
        for m in 1:S2
            fi = Jm.pixels[m, a + 1]
            fi != 0 || continue
            acc = zero(eltype(y))
            for k in 1:p
                acc += Jm.values[k, m, a + 1] * v[base + k]
            end
            y[fi] += acc
        end
    end
    return y
end

# ==============================================================================
# Neighbor model (the sparsity pattern of H)
# ==============================================================================

# Off-diagonal p x p blocks of H, stored flat (CSR-style) rather than as one
# heap object per star-pair.  At high crowding the pair count can reach into
# the millions, so a `Vector` of small mutable structs, each separately
# allocating its own `Matrix` and two `Vector`s, multiplies both memory and
# allocation count far beyond the raw data size and makes `_accumulate_H!`/
# `_apply_H!` pointer-chase through scattered heap objects instead of walking
# contiguous arrays.  `pair_a[t] > pair_b[t]` always (lower triangle); pair
# `t`'s shared stamp-pixel offsets are `shared_ma[offsets[t]:offsets[t+1]-1]`
# (and `shared_mb` likewise).
struct NeighborBlocks{T}
    pair_a::Vector{Int32}
    pair_b::Vector{Int32}
    offsets::Vector{Int}
    shared_ma::Vector{Int32}
    shared_mb::Vector{Int32}
    B::Array{T, 3}
end

Base.length(nb::NeighborBlocks) = length(nb.pair_a)

# ---------------------------------------------------------------------------
# Build the fixed stamp footprint and the neighbor structure once.
# ---------------------------------------------------------------------------

function _build_stamps!(image, params, row_y, row_x, row_flux, fit_rad, w, FT)
    ny, nx = size(image)
    n_stars = size(params, 2)

    R = round(Int, round(fit_rad, RoundUp))
    S = 2R + 1
    S2 = S * S

    dy_off = Vector{Int}(undef, S2)
    dx_off = Vector{Int}(undef, S2)
    m = 0
    for dx in -R:R, dy in -R:R
        m += 1
        dy_off[m] = dy
        dx_off[m] = dx
    end

    anchor_y = round.(Int, view(params, row_y, :))
    anchor_x = round.(Int, view(params, row_x, :))

    pixels = zeros(Int32, S2, n_stars)
    for i in 1:n_stars
        ay = anchor_y[i]
        ax = anchor_x[i]
        for m in 1:S2
            gy = ay + dy_off[m]
            gx = ax + dx_off[m]
            if 1 <= gy <= ny && 1 <= gx <= nx
                fi = gy + (gx - 1) * ny
                if w[fi] > 0
                    pixels[m, i] = fi
                end
            end
        end
    end

    # Active stars: at least one usable pixel and finite initial y/x/flux.
    active = Int[]
    valid = trues(n_stars)
    failure_msgs = String[]
    for i in 1:n_stars
        ok = any(!iszero, view(pixels, :, i)) &&
             isfinite(params[row_y, i]) && isfinite(params[row_x, i]) &&
             isfinite(params[row_flux, i])
        if ok
            push!(active, i)
        else
            valid[i] = false
            push!(failure_msgs, "star $i: excluded before the solve (no usable pixels or non-finite initial parameters)")
        end
    end

    n_active = length(active)
    pixels_act = zeros(Int32, S2, n_active)
    anchor_y_act = Vector{Int}(undef, n_active)
    anchor_x_act = Vector{Int}(undef, n_active)
    for (j, i) in enumerate(active)
        pixels_act[:, j] .= view(pixels, :, i)
        anchor_y_act[j] = anchor_y[i]
        anchor_x_act[j] = anchor_x[i]
    end

    return (
        pixels = pixels_act, anchor_y = anchor_y_act, anchor_x = anchor_x_act,
        dy_off = dy_off, dx_off = dx_off, active = active, valid = valid,
        n_failed = n_stars - n_active, failure_msgs = failure_msgs, S2 = S2,
    )
end

function _build_neighbors(pixels, S2, n_active, p, FT)
    # Count usable pixels.
    nrec = 0
    @inbounds for a in 1:n_active, m in 1:S2
        pixels[m, a] != 0 && (nrec += 1)
    end
    rec_pix = Vector{Int}(undef, nrec)
    rec_star = Vector{Int32}(undef, nrec)
    rec_off = Vector{Int32}(undef, nrec)
    k = 0
    @inbounds for a in 1:n_active, m in 1:S2
        fi = pixels[m, a]
        fi != 0 || continue
        k += 1
        rec_pix[k] = fi
        rec_star[k] = a - 1
        rec_off[k] = m - 1
    end

    perm = sortperm(rec_pix)

    # Pass 1: count union pixels and total raw (unordered star-pair, shared
    # pixel) records, so pass 2 can fill preallocated arrays instead of
    # growing them with `push!`, which has to dynamically resize the array
    n_union = 0
    total_shared = 0
    i = 1
    n = nrec
    while i <= n
        j = i
        while j <= n && rec_pix[perm[j]] == rec_pix[perm[i]]
            j += 1
        end
        len = j - i
        n_union += 1
        total_shared += (len * (len - 1)) ÷ 2
        i = j
    end

    union_pix = Vector{Int}(undef, n_union)
    pair_a = Vector{Int32}(undef, total_shared)
    pair_b = Vector{Int32}(undef, total_shared)
    pair_ma = Vector{Int32}(undef, total_shared)
    pair_mb = Vector{Int32}(undef, total_shared)

    # Pass 2: fill union pixels and raw pair records.
    ui = 0
    t = 0
    i = 1
    while i <= n
        j = i
        while j <= n && rec_pix[perm[j]] == rec_pix[perm[i]]
            j += 1
        end
        ui += 1
        union_pix[ui] = rec_pix[perm[i]]
        len = j - i
        @inbounds for u in 1:len
            su = rec_star[perm[i + u - 1]]
            mu = rec_off[perm[i + u - 1]]
            for v in (u + 1):len
                sv = rec_star[perm[i + v - 1]]
                mv = rec_off[perm[i + v - 1]]
                t += 1
                if su > sv
                    pair_a[t] = su; pair_b[t] = sv
                    pair_ma[t] = mu; pair_mb[t] = mv
                else
                    pair_a[t] = sv; pair_b[t] = su
                    pair_ma[t] = mv; pair_mb[t] = mu
                end
            end
        end
        i = j
    end

    if total_shared == 0
        return union_pix, NeighborBlocks{FT}(
            Int32[], Int32[], Int[], Int32[], Int32[], zeros(FT, p, p, 0))
    end

    # Group raw records by (a, b) via one sort: equal keys are contiguous
    # after sorting, so the sorted shared-pixel arrays ARE the flat per-pair
    # storage directly -- no further `push!`-based grouping into per-pair
    # containers is needed.
    key = Vector{Int64}(undef, total_shared)
    @inbounds for t in 1:total_shared
        key[t] = Int64(pair_a[t]) * n_active + pair_b[t]
    end
    ord = sortperm(key)
    shared_ma = pair_ma[ord]
    shared_mb = pair_mb[ord]
    sorted_key = key[ord]

    npair = 1
    @inbounds for t in 2:total_shared
        sorted_key[t] != sorted_key[t - 1] && (npair += 1)
    end

    out_a = Vector{Int32}(undef, npair)
    out_b = Vector{Int32}(undef, npair)
    offsets = Vector{Int}(undef, npair + 1)
    bi = 0
    @inbounds for t in 1:total_shared
        if t == 1 || sorted_key[t] != sorted_key[t - 1]
            bi += 1
            offsets[bi] = t
            out_a[bi] = pair_a[ord[t]]
            out_b[bi] = pair_b[ord[t]]
        end
    end
    offsets[npair + 1] = total_shared + 1

    B = zeros(FT, p, p, npair)
    return union_pix, NeighborBlocks{FT}(out_a, out_b, offsets, shared_ma, shared_mb, B)
end

# ==============================================================================
# Fill (render value + gradient), H accumulation, matvec
# ==============================================================================

function _fill_stamps!(
        stamp::StampDerivatives{FT}, model_template, free_names_val, fixed,
        θ, w, model_img, grad_col, dy_off, dx_off, anchor_y, anchor_x,
        row_y, row_x, row_flux, live, _fill_scratch_buf
    ) where {FT}
    p = stamp.p
    S2 = stamp.S2
    n_active = size(stamp.values, 3)
    fill!(model_img, zero(FT))
    # No blanket `fill!(stamp.colnorm, ...)`: a frozen star's column must stay
    # bitwise untouched between freeze and the end of the fit (reset only the
    # live columns being recomputed below), or the eps(FT) floor two lines
    # down turns a stale zero into a ~4.5e15 rescale applied again every
    # iteration, overflowing stamp.values to Inf within a couple of outer
    # iterations and leaking NaN into a still-live neighbor's H block.
    @inbounds for a in 0:(n_active - 1)
        live[a + 1] || continue
        base = a * p
        for k in 1:p
            stamp.colnorm[k, a + 1] = zero(FT)
        end
        m = PSF.model_from_vector(model_template, free_names_val, view(θ, base + 1:base + p), fixed)
        ay = anchor_y[a + 1]
        ax = anchor_x[a + 1]
        for mi in 1:S2
            fi = stamp.pixels[mi, a + 1]
            fi != 0 || continue
            gy = ay + dy_off[mi]
            gx = ax + dx_off[mi]
            f, g = evaluate_fg(m, gy, gx)
            g1, g2, g3 = g[row_y], g[row_x], g[row_flux]
            model_img[fi] += f
            sw = sqrt(w[fi])
            for k in 1:p
                gc = grad_col[k]
                gv = gc == 1 ? g1 : (gc == 2 ? g2 : g3)
                raw = sw * gv
                stamp.values[k, mi, a + 1] = raw
                stamp.colnorm[k, a + 1] += raw * raw
            end
        end
    end
    # Column equilibration: values ./= colnorm, floor colnorm at eps(T).
    @inbounds for a in 1:n_active
        live[a] || continue
        for k in 1:p
            cn = sqrt(stamp.colnorm[k, a])
            cn = ifelse(cn < eps(FT), eps(FT), cn)
            stamp.colnorm[k, a] = cn
            invcn = inv(cn)
            for mi in 1:S2
                stamp.values[k, mi, a] *= invcn
            end
        end
    end
    return stamp
end

"""
    _fill_scratch(psf, S, ::Type{FT}) where {FT}

Per-solve scratch for [`_fill_stamps!`](@ref)'s specialized
`GriddedPSFModel{T,<:ImagePSF{T}}` method: the same per-corner
`(value, d/dv, d/du)` `S x S` buffers `_render_scratch` builds, plus three
length-`S^2` reduction outputs (`dS/dY`, `dS/dX`, `S`, in
`evaluate_fg(::GriddedPSFModel,...)`'s notation). `nothing` for every other
model type, which the generic `_fill_stamps!` method ignores. Kept
independent of `_render_scratch`'s buffers (rather than shared) even though
`_fill_stamps!` and `_render_model!` never run concurrently within one
solve -- the tiny extra allocation is not worth the coupling.
"""
_fill_scratch(::AbstractPSFModel, S::Int, ::Type{FT}) where {FT} = nothing
function _fill_scratch(::GriddedPSFModel{T2, M}, S::Int, ::Type{FT}) where {T2, M <: ImagePSF{T2}, FT}
    corner = ntuple(_ -> ntuple(_ -> Matrix{FT}(undef, S, S), 4), 3)
    reductions = ntuple(_ -> Vector{FT}(undef, S * S), 3)
    return corner, reductions
end

"""
    _fill_stamps!(stamp, model_template::GriddedPSFModel{T,<:ImagePSF{T}}, ...)

Specialized `LV.@turbo` value+gradient fill for a
`GriddedPSFModel{T,<:ImagePSF{T}}` PSF, mirroring the specialized
[`_render_model!`](@ref): corner selection/weights and each active node's
`(oversampling, origin, fill_value)` depend only on the star's `(Y, X)`, so
they are computed once per star rather than once per SIMD batch. The actual
bicubic gather reuses [`PSF._gridded_corner_bicubic_pass!`](@ref) (one
branchless `@turbo` pass per corner, already used by the sequential
`fit_star` path), and a `@turbo` reduction combines the four corners' value
and both partial derivatives into `S`, `dS/dY`, `dS/dX` -- the same chain
rule `evaluate_fg(::GriddedPSFModel,...)` and `_accum_gridded_imagepsf!` use
-- before the per-pixel masked accumulation into `stamp.values`/`colnorm`
proceeds exactly as the generic method's.
"""
function _fill_stamps!(
        stamp::StampDerivatives{FT}, model_template::GriddedPSFModel{T2, M}, free_names_val, fixed,
        θ, w, model_img, grad_col, dy_off, dx_off, anchor_y, anchor_x,
        row_y, row_x, row_flux, live,
        fill_scratch_buf::Tuple{NTuple{3, NTuple{4, Matrix{FT}}}, NTuple{3, Vector{FT}}}
    ) where {FT, T2, M <: ImagePSF{T2}}
    p = stamp.p
    S2 = stamp.S2
    S = round(Int, sqrt(S2))
    R = maximum(dy_off)
    n_active = size(stamp.values, 3)
    fill!(model_img, zero(FT))
    (p_val, p_dpdv, p_dpdu), (dsdY_buf, dsdX_buf, s_buf) = fill_scratch_buf
    pv1, pv2, pv3, pv4 = p_val
    pdv1, pdv2, pdv3, pdv4 = p_dpdv
    pdu1, pdu2, pdu3, pdu4 = p_dpdu
    dsdY_mat = reshape(dsdY_buf, S, S)
    dsdX_mat = reshape(dsdX_buf, S, S)
    s_mat = reshape(s_buf, S, S)
    # No blanket `fill!(stamp.colnorm, ...)` -- see the generic method's comment.
    @inbounds for a in 0:(n_active - 1)
        live[a + 1] || continue
        base = a * p
        for k in 1:p
            stamp.colnorm[k, a + 1] = zero(FT)
        end
        m = PSF.model_from_vector(model_template, free_names_val, view(θ, base + 1:base + p), fixed)
        Y, X, flux, bkg = FT(m.y), FT(m.x), FT(m.flux), FT(m.bkg)
        ay = anchor_y[a + 1]
        ax = anchor_x[a + 1]
        yr = (ay - R):(ay + R)
        xr = (ax - R):(ax + R)

        corners = PSF._grid_corners_dw(model_template, Y, X)
        idx1, w1, dwdy1, dwdx1 = corners[1]
        idx2, w2, dwdy2, dwdx2 = corners[2]
        idx3, w3, dwdy3, dwdx3 = corners[3]
        idx4, w4, dwdy4, dwdx4 = corners[4]
        idx2 = idx2 == 0 ? idx1 : idx2
        idx3 = idx3 == 0 ? idx1 : idx3
        idx4 = idx4 == 0 ? idx1 : idx4
        w1, w2, w3, w4 = FT(w1), FT(w2), FT(w3), FT(w4)
        dwdy1, dwdy2, dwdy3, dwdy4 = FT(dwdy1), FT(dwdy2), FT(dwdy3), FT(dwdy4)
        dwdx1, dwdx2, dwdx3, dwdx4 = FT(dwdx1), FT(dwdx2), FT(dwdx3), FT(dwdx4)
        node1, node2, node3, node4 = model_template.psfs[idx1], model_template.psfs[idx2],
            model_template.psfs[idx3], model_template.psfs[idx4]
        sx1, sy1 = FT(node1.oversampling[1]), FT(node1.oversampling[2])
        sx2, sy2 = FT(node2.oversampling[1]), FT(node2.oversampling[2])
        sx3, sy3 = FT(node3.oversampling[1]), FT(node3.oversampling[2])
        sx4, sy4 = FT(node4.oversampling[1]), FT(node4.oversampling[2])

        y1, x1 = ay - R, ax - R
        PSF._gridded_corner_bicubic_pass!(pv1, pdv1, pdu1, node1.data, FT(node1.origin.x), FT(node1.origin.y),
            sx1, sy1, FT(node1.fill_value), size(node1.data, 1), size(node1.data, 2), yr, xr, Y, X, y1, x1)
        PSF._gridded_corner_bicubic_pass!(pv2, pdv2, pdu2, node2.data, FT(node2.origin.x), FT(node2.origin.y),
            sx2, sy2, FT(node2.fill_value), size(node2.data, 1), size(node2.data, 2), yr, xr, Y, X, y1, x1)
        PSF._gridded_corner_bicubic_pass!(pv3, pdv3, pdu3, node3.data, FT(node3.origin.x), FT(node3.origin.y),
            sx3, sy3, FT(node3.fill_value), size(node3.data, 1), size(node3.data, 2), yr, xr, Y, X, y1, x1)
        PSF._gridded_corner_bicubic_pass!(pv4, pdv4, pdu4, node4.data, FT(node4.origin.x), FT(node4.origin.y),
            sx4, sy4, FT(node4.fill_value), size(node4.data, 1), size(node4.data, 2), yr, xr, Y, X, y1, x1)

        # Chain rule reduction combining the four corners -- matches
        # `evaluate_fg(::GriddedPSFModel,...)` / `_accum_gridded_imagepsf!`.
        LV.@turbo for jj in 1:S, ii in 1:S
            s_mat[ii, jj] = w1 * pv1[ii, jj] + w2 * pv2[ii, jj] + w3 * pv3[ii, jj] + w4 * pv4[ii, jj]
            dsdY_mat[ii, jj] = dwdy1 * pv1[ii, jj] - w1 * sy1 * pdv1[ii, jj] +
                dwdy2 * pv2[ii, jj] - w2 * sy2 * pdv2[ii, jj] +
                dwdy3 * pv3[ii, jj] - w3 * sy3 * pdv3[ii, jj] +
                dwdy4 * pv4[ii, jj] - w4 * sy4 * pdv4[ii, jj]
            dsdX_mat[ii, jj] = dwdx1 * pv1[ii, jj] - w1 * sx1 * pdu1[ii, jj] +
                dwdx2 * pv2[ii, jj] - w2 * sx2 * pdu2[ii, jj] +
                dwdx3 * pv3[ii, jj] - w3 * sx3 * pdu3[ii, jj] +
                dwdx4 * pv4[ii, jj] - w4 * sx4 * pdu4[ii, jj]
        end

        for mi in 1:S2
            fi = stamp.pixels[mi, a + 1]
            fi != 0 || continue
            s_val = s_buf[mi]
            f = muladd(flux, s_val, bkg)
            gy = flux * dsdY_buf[mi]
            gx = flux * dsdX_buf[mi]
            gflux = s_val
            model_img[fi] += f
            sw = sqrt(w[fi])
            for k in 1:p
                gc = grad_col[k]
                gv = gc == 1 ? gy : (gc == 2 ? gx : gflux)
                raw = sw * gv
                stamp.values[k, mi, a + 1] = raw
                stamp.colnorm[k, a + 1] += raw * raw
            end
        end
    end
    # Column equilibration: values ./= colnorm, floor colnorm at eps(T).
    @inbounds for a in 1:n_active
        live[a] || continue
        for k in 1:p
            cn = sqrt(stamp.colnorm[k, a])
            cn = ifelse(cn < eps(FT), eps(FT), cn)
            stamp.colnorm[k, a] = cn
            invcn = inv(cn)
            for mi in 1:S2
                stamp.values[k, mi, a] *= invcn
            end
        end
    end
    return stamp
end

function _render_model!(
        model_img::AbstractVector{FT}, model_template, free_names_val, fixed,
        θ, p, dy_off, dx_off, anchor_y, anchor_x, pixels, live, render_buf::AbstractVector{FT},
        _corner_scratch
    ) where {FT}
    fill!(model_img, zero(FT))
    S2 = length(dy_off)
    n_active = length(anchor_y)
    @inbounds for a in 0:(n_active - 1)
        live[a + 1] || continue
        base = a * p
        m = PSF.model_from_vector(model_template, free_names_val, view(θ, base + 1:base + p), fixed)
        ay = anchor_y[a + 1]
        ax = anchor_x[a + 1]
        # `evaluate` is declared `LV.can_turbo`-safe for every PSF model
        # (PSF.jl), so this per-pixel render vectorizes -- but an `LV.@turbo`
        # loop cannot early-exit (`fi != 0 || continue`, for off-image/masked
        # stamp positions) and cannot safely scatter into the shared
        # `model_img` (different stars' stamps can alias the same pixel).
        # Render every offset (masked or not -- wasted but harmless, since a
        # masked value is simply never scattered below) into a dense,
        # non-aliased per-star scratch buffer, then mask-and-accumulate as a
        # separate, cheap plain loop.
        LV.@turbo for mi in 1:S2
            render_buf[mi] = evaluate(m, ay + dy_off[mi], ax + dx_off[mi])
        end
        for mi in 1:S2
            fi = pixels[mi, a + 1]
            fi != 0 || continue
            model_img[fi] += render_buf[mi]
        end
    end
    return model_img
end

"""
    _render_scratch(psf, S, ::Type{FT}) where {FT}

Per-solve scratch for [`_render_model!`](@ref)'s specialized
`GriddedPSFModel{T,<:ImagePSF{T}}` method: three `NTuple{4,Matrix{FT}}` (value,
d/dv, d/du), each corner's own `S x S` stamp-shaped buffer. `nothing` for
every other model type, which the generic `_render_model!` method ignores.
"""
_render_scratch(::AbstractPSFModel, S::Int, ::Type{FT}) where {FT} = nothing
function _render_scratch(::GriddedPSFModel{T2, M}, S::Int, ::Type{FT}) where {T2, M <: ImagePSF{T2}, FT}
    return ntuple(_ -> ntuple(_ -> Matrix{FT}(undef, S, S), 4), 3)
end

"""
    _render_model!(model_img, model_template::GriddedPSFModel{T,<:ImagePSF{T}}, ...)

Specialized renderer for`GriddedPSFModel{T,<:ImagePSF{T}}` PSF:
avoids the generic method's per-`evaluate` call recomputation of corner
selection/weights. Corner indices/weights and each active node's recentered
origin depend only on the star's `(Y, X)`, not on which pixel is being
evaluated, so they are computed once per star here; the actual bicubic
gather+blend reuses [`PSF._gridded_corner_bicubic_pass!`](@ref)
(the same kernel the sequential `fit_star` path uses),
one branchless pass per corner into `corner_scratch`'s per-corner `S x S`
buffers, followed by a weighted-sum reduction into `render_buf`.
"""
function _render_model!(
        model_img::AbstractVector{FT}, model_template::GriddedPSFModel{T2, M}, free_names_val, fixed,
        θ, p, dy_off, dx_off, anchor_y, anchor_x, pixels, live, render_buf::AbstractVector{FT},
        corner_scratch::NTuple{3, NTuple{4, Matrix{FT}}}
    ) where {FT, T2, M <: ImagePSF{T2}}
    fill!(model_img, zero(FT))
    S2 = length(dy_off)
    S = round(Int, sqrt(S2))
    R = maximum(dy_off)
    n_active = length(anchor_y)
    p_val, p_dpdv, p_dpdu = corner_scratch
    pv1, pv2, pv3, pv4 = p_val
    pdv1, pdv2, pdv3, pdv4 = p_dpdv
    pdu1, pdu2, pdu3, pdu4 = p_dpdu
    render_mat = reshape(render_buf, S, S)
    @inbounds for a in 0:(n_active - 1)
        live[a + 1] || continue
        base = a * p
        m = PSF.model_from_vector(model_template, free_names_val, view(θ, base + 1:base + p), fixed)
        Y, X, flux, bkg = FT(m.y), FT(m.x), FT(m.flux), FT(m.bkg)
        ay = anchor_y[a + 1]
        ax = anchor_x[a + 1]
        yr = (ay - R):(ay + R)
        xr = (ax - R):(ax + R)

        # Corner selection/weights and node lookup: loop-invariant across the
        # whole stamp (depend only on Y, X), so this runs once per star, not
        # once per SIMD batch as the generic (opaque-call) method does.
        corners = PSF._grid_corners_dw(model_template, Y, X)
        idx1, w1 = corners[1][1], FT(corners[1][2])
        idx2, w2 = corners[2][1], FT(corners[2][2])
        idx3, w3 = corners[3][1], FT(corners[3][2])
        idx4, w4 = corners[4][1], FT(corners[4][2])
        # `_grid_corners_dw` returns idx=0 for corners 2-4 only when there is
        # a single node; their weight is exactly 0 there, so remapping to a
        # valid (arbitrary) index is safe (matches `_accum_gridded_imagepsf!`).
        idx2 = idx2 == 0 ? idx1 : idx2
        idx3 = idx3 == 0 ? idx1 : idx3
        idx4 = idx4 == 0 ? idx1 : idx4
        node1, node2, node3, node4 = model_template.psfs[idx1], model_template.psfs[idx2],
            model_template.psfs[idx3], model_template.psfs[idx4]

        y1, x1 = ay - R, ax - R
        PSF._gridded_corner_bicubic_pass!(pv1, pdv1, pdu1, node1.data, FT(node1.origin.x), FT(node1.origin.y),
            FT(node1.oversampling[1]), FT(node1.oversampling[2]), FT(node1.fill_value),
            size(node1.data, 1), size(node1.data, 2), yr, xr, Y, X, y1, x1)
        PSF._gridded_corner_bicubic_pass!(pv2, pdv2, pdu2, node2.data, FT(node2.origin.x), FT(node2.origin.y),
            FT(node2.oversampling[1]), FT(node2.oversampling[2]), FT(node2.fill_value),
            size(node2.data, 1), size(node2.data, 2), yr, xr, Y, X, y1, x1)
        PSF._gridded_corner_bicubic_pass!(pv3, pdv3, pdu3, node3.data, FT(node3.origin.x), FT(node3.origin.y),
            FT(node3.oversampling[1]), FT(node3.oversampling[2]), FT(node3.fill_value),
            size(node3.data, 1), size(node3.data, 2), yr, xr, Y, X, y1, x1)
        PSF._gridded_corner_bicubic_pass!(pv4, pdv4, pdu4, node4.data, FT(node4.origin.x), FT(node4.origin.y),
            FT(node4.oversampling[1]), FT(node4.oversampling[2]), FT(node4.fill_value),
            size(node4.data, 1), size(node4.data, 2), yr, xr, Y, X, y1, x1)

        LV.@turbo for jj in 1:S, ii in 1:S
            render_mat[ii, jj] = muladd(flux, w1 * pv1[ii, jj] + w2 * pv2[ii, jj] + w3 * pv3[ii, jj] + w4 * pv4[ii, jj], bkg)
        end
        for mi in 1:S2
            fi = pixels[mi, a + 1]
            fi != 0 || continue
            model_img[fi] += render_buf[mi]
        end
    end
    return model_img
end

function _residual_cost!(wt_resid, model_img, data, w, union_pix)
    cost = zero(eltype(data))
    @inbounds for q in union_pix
        r = model_img[q] - data[q]
        wr = sqrt(w[q]) * r
        wt_resid[q] = wr
        cost += wr * wr
    end
    return cost
end

function _cost!(model_img, data, w, union_pix)
    cost = zero(eltype(data))
    @inbounds for q in union_pix
        r = model_img[q] - data[q]
        cost += w[q] * r * r
    end
    return cost
end

"""
    _accumulate_H!(Hdiag, nb, stamp)

Fill the diagonal blocks `Hdiag` and the off-diagonal blocks `nb.B` from the
current `stamp` values.

Both loops are written to vectorize under `LV.@turbo`:

- The diagonal-block loop drops the `stamp.pixels[m, a] != 0` mask check
  present in earlier versions. It is provably redundant, not just
  unnecessary: `_fill_stamps!` only ever writes `stamp.values[:, m, a]` at
  pixels with `stamp.pixels[m, a] != 0`, its column-equilibration pass
  multiplies every (including masked) entry by a finite scale, and
  `stamp.values` starts zero-filled at construction -- so masked entries are
  `0` for the entire fit and contribute `0` to the sum whether or not they
  are skipped. A `continue`-based skip is also, independently, not legal
  inside a `@turbo` (or even a plain `@simd`) loop body.
- The off-diagonal loop cannot be one flat `@turbo` block: different pairs
  have different-length shared-pixel runs (`nb.offsets`), and every shared
  pixel in a pair's run must accumulate into that *same* pair's `p x p`
  block, which rules out treating the whole flat `shared_ma`/`shared_mb`
  array as one independent-iteration reduction. Instead each pair gets its
  own `@turbo` block over `(s, k, l)`, gathering `stamp.values` through the
  `shared_ma`/`shared_mb` offsets.

Measured together, this cut `_accumulate_H!`'s wall time by `~3.2x` on an
otherwise-identical fit (`N = 10,000` active stars, `fit_rad = 5`) with
bitwise-irrelevant differences from the scalar path (`rtol = 1e-10`).
"""
function _accumulate_H!(Hdiag, nb::NeighborBlocks, stamp::StampDerivatives)
    p = stamp.p
    S2 = stamp.S2
    n_active = size(stamp.values, 3)
    values = stamp.values
    fill!(Hdiag, zero(eltype(Hdiag)))
    LV.@turbo for a in 1:n_active, m in 1:S2, k in 1:p, l in 1:p
        Hdiag[k, l, a] += values[k, m, a] * values[l, m, a]
    end

    npair = length(nb)
    pair_a = nb.pair_a
    pair_b = nb.pair_b
    offsets = nb.offsets
    shared_ma = nb.shared_ma
    shared_mb = nb.shared_mb
    B = nb.B
    @inbounds for t in 1:npair
        a = pair_a[t] + 1
        b = pair_b[t] + 1
        lo = offsets[t]
        hi = offsets[t + 1] - 1
        @inbounds for k in 1:p, l in 1:p
            B[k, l, t] = zero(eltype(B))
        end
        LV.@turbo for s in lo:hi, k in 1:p, l in 1:p
            ma = shared_ma[s] + Int32(1)
            mb = shared_mb[s] + Int32(1)
            B[k, l, t] += values[k, ma, a] * values[l, mb, b]
        end
    end
    return Hdiag
end

function _apply_H!(y, Hdiag, nb::NeighborBlocks, x)
    p = size(Hdiag, 1)
    n_active = size(Hdiag, 3)
    fill!(y, zero(eltype(y)))
    @inbounds for a in 0:(n_active - 1)
        base = a * p
        for k in 1:p
            acc = zero(eltype(y))
            for l in 1:p
                acc += Hdiag[k, l, a + 1] * x[base + l]
            end
            y[base + k] += acc
        end
    end
    npair = length(nb)
    @inbounds for t in 1:npair
        a = nb.pair_a[t]
        b = nb.pair_b[t]
        ba = a * p
        bb = b * p
        for k in 1:p
            acc_row = zero(eltype(y))
            acc_col = zero(eltype(y))
            for l in 1:p
                acc_row += nb.B[k, l, t] * x[bb + l]
                acc_col += nb.B[l, k, t] * x[ba + l]
            end
            y[ba + k] += acc_row
            y[bb + k] += acc_col
        end
    end
    return y
end

function _apply_A!(y, Hdiag, nbr_blocks, λ, x)
    _apply_H!(y, Hdiag, nbr_blocks, x)
    @. y += λ * x
    return y
end

# ==============================================================================
# Block-Jacobi preconditioner and preconditioned CG
# ==============================================================================

function _build_precond!(Mblocks, Hdiag, λ)
    p, _, n_active = size(Hdiag)
    FT = eltype(Hdiag)
    @inbounds for a in 1:n_active
        if p == 3
            a11 = Hdiag[1, 1, a] + λ
            a12 = Hdiag[1, 2, a]
            a13 = Hdiag[1, 3, a]
            a22 = Hdiag[2, 2, a] + λ
            a23 = Hdiag[2, 3, a]
            a33 = Hdiag[3, 3, a] + λ
            Minv = inv(@SMatrix [a11 a12 a13; a12 a22 a23; a13 a23 a33])
            Mblocks[1, 1, a] = Minv[1, 1]; Mblocks[1, 2, a] = Minv[1, 2]; Mblocks[1, 3, a] = Minv[1, 3]
            Mblocks[2, 1, a] = Minv[2, 1]; Mblocks[2, 2, a] = Minv[2, 2]; Mblocks[2, 3, a] = Minv[2, 3]
            Mblocks[3, 1, a] = Minv[3, 1]; Mblocks[3, 2, a] = Minv[3, 2]; Mblocks[3, 3, a] = Minv[3, 3]
        else
            blk = zeros(FT, p, p)
            for k in 1:p, l in 1:p
                blk[k, l] = Hdiag[k, l, a]
            end
            for k in 1:p
                blk[k, k] += λ
            end
            Minv = inv(blk)
            for k in 1:p, l in 1:p
                Mblocks[k, l, a] = Minv[k, l]
            end
        end
    end
    return Mblocks
end

function _apply_precond!(z, Mblocks, r)
    p, _, n_active = size(Mblocks)
    @inbounds for a in 0:(n_active - 1)
        base = a * p
        for k in 1:p
            acc = zero(eltype(z))
            for l in 1:p
                acc += Mblocks[k, l, a + 1] * r[base + l]
            end
            z[base + k] = acc
        end
    end
    return z
end

"""
    _mask_frozen!(v, frozen_list, p)

Zero every frozen star's `p`-slice of `v` in place. `frozen_list` holds
`0`-based star indices; used inside [`_pcg!`](@ref) after every `_apply_A!`
call so that a still-live neighbor's search direction can never leak a
nonzero value into a frozen star's slot within the current solve (see the
correctness analysis in `_pcg!`'s docstring).
"""
function _mask_frozen!(v, frozen_list, p)
    @inbounds for s in frozen_list
        base = s * p
        for k in 1:p
            v[base + k] = zero(eltype(v))
        end
    end
    return v
end

"""
    _pcg!(δ, Hdiag, nbr_blocks, λ, rhs, Mblocks, inner_iterations, r, z, pk, Ap, frozen_list, p, cg_tol)

Block-Jacobi preconditioned CG solve of `(H + λI) δ = rhs`.

`frozen_list` (`0`-based star indices no longer being fit) is used to zero
`Ap`'s frozen-star slices immediately after every [`_apply_A!`](@ref) call.
This is required for correctness, not just cheap insurance: `_apply_H!`'s
off-diagonal term writes `y[frozen] += B' * x[active]` for any pair still
touching a frozen star, so even with `rhs[frozen] == 0` (guaranteed by
[`apply_JT!`](@ref) skipping frozen stars) a still-live neighbor's nonzero
search direction would otherwise leak into `Ap[frozen]` on the very first
`_apply_A!` call, then propagate into `r`/`z`/`pk` on later CG iterations,
leaving a "frozen" star with a nonzero `δ` at the end of the solve. Masking
`Ap` every iteration keeps `pk`/`r`/`z`/`δ` at frozen slices at exactly `0`
by induction, so CG runs exactly on the reduced (frozen coordinates
projected out) system rather than an approximation of it.

`inner_iterations` is an upper cap; the loop also stops early as soon as the
true (unpreconditioned) relative residual `‖rhs − Aδ‖ / ‖rhs‖` drops to
`cg_tol` or below. This only ever *shortens* a solve relative to running the
full cap, so it is a safe, accuracy-preserving performance win for the
common case: an isolated or lightly-blended star's system is solved exactly
in 1-2 iterations by the block-Jacobi preconditioner, and the loop no longer
burns the rest of the cap on a converged answer.

Do **not** treat `cg_tol`/`inner_iterations` as a knob for fixing bright-star
precision in crowded fields by making the solve *more* exact -- raising
`inner_iterations` (or tightening `cg_tol`) beyond what the default already
resolves does not self-correct a genuinely degenerate, tightly-blended group
of stars; it actively destabilizes it. A star's true position/flux update in
a near-null-space direction (e.g. two heavily overlapping stars whose fluxes
are almost exchangeable at fixed total light) is only weakly determined by
the data, and CG resolves that ill-conditioned direction more and more
precisely the more iterations it is given -- amplifying noise into large,
sometimes negative, flux swings instead of converging to a better answer.
Nothing in the outer Levenberg-Marquardt loop guards against this: `λ` and
step acceptance are driven by the *global* cost across all stars, so a step
that blows up one small degenerate group can still be accepted because the
much larger well-behaved population dominates the sum. A full sweep of
`cg_tol` at `inner_iterations = 50` on a realistic crowded field (see
`experiments/simul-solve/`) showed a monotonic trade: tighter `cg_tol`
buys smaller bright-star bias but a proportionally larger population of
stars whose flux diverges to a non-positive value, with no tolerance that
improves one without worsening the other. The fix for that failure mode
belongs in the damping/acceptance logic (something that protects a
degenerate subgroup specifically), not in the CG stopping rule.
"""
function _pcg!(δ, Hdiag, nbr_blocks, λ, rhs, Mblocks, inner_iterations, r, z, pk, Ap, frozen_list, p, cg_tol)
    FT = eltype(δ)
    fill!(δ, zero(FT))
    r .= rhs
    rhs_norm = norm(rhs)
    rhs_norm == 0 && return δ
    _apply_precond!(z, Mblocks, r)
    pk .= z
    rz = dot(r, z)
    rz == 0 && return δ
    for _ in 1:inner_iterations
        _apply_A!(Ap, Hdiag, nbr_blocks, λ, pk)
        _mask_frozen!(Ap, frozen_list, p)
        pap = dot(pk, Ap)
        pap == 0 && break
        α = rz / pap
        @. δ += α * pk
        @. r -= α * Ap
        norm(r) <= FT(cg_tol) * rhs_norm && break
        _apply_precond!(z, Mblocks, r)
        rz_new = dot(r, z)
        rz_new == 0 && break
        β = rz_new / rz
        @. pk = z + β * pk
        rz = rz_new
    end
    return δ
end

"""
    CholeskySolverCache{T}

Cached CHOLMOD state for the `:cholesky` oracle solver.  The sparsity
pattern of `H`'s lower triangle is fixed for the whole fit (§4.2's fixed
stamp anchor), so it is built once from the neighbor structure; only the
numeric values change every outer iteration.  `diag_pos`/`pair_pos` map each
logical `Hdiag`/`nb.B` entry to its position in the *unsorted* triplet list
built at cache-construction time, and `invord` maps that triplet position to
its final slot in `H.nzval` -- together they let [`_refill_cholesky!`](@ref)
write refreshed values directly into `H.nzval` without rebuilding the sparse
structure. `F` holds the CHOLMOD factorization, built lazily on first use and
thereafter reused via `cholesky!` (reanalyzing only the numeric factorization,
not the symbolic structure) as `λ` and the values change from iteration to
iteration and trial to trial.
"""
mutable struct CholeskySolverCache{T}
    H::SparseArrays.SparseMatrixCSC{T, Int}
    diag_pos::Array{Int, 3}
    pair_pos::Array{Int, 3}
    invord::Vector{Int}
    F::Any
end

function _build_cholesky_cache(nb::NeighborBlocks{FT}, n_active::Int, p::Int) where {FT}
    n = p * n_active
    npair = length(nb)
    ndiag = n_active * (p * (p + 1)) ÷ 2
    ntot = ndiag + npair * p * p

    rows = Vector{Int}(undef, ntot)
    cols = Vector{Int}(undef, ntot)
    diag_pos = zeros(Int, p, p, n_active)
    pair_pos = zeros(Int, p, p, npair)

    # Only the lower triangle (k >= l) of each diagonal block is needed: it
    # is symmetric by construction, and `Symmetric(H, :L)` mirrors the rest.
    # Every off-diagonal pair block lies entirely below the global diagonal
    # (pair_a[t] > pair_b[t] guarantees ba + k > bb + l for all k, l in
    # 1:p), so the full p x p block is stored for pairs.
    t = 0
    @inbounds for a in 0:(n_active - 1)
        base = a * p
        for k in 1:p, l in 1:k
            t += 1
            rows[t] = base + k
            cols[t] = base + l
            diag_pos[k, l, a + 1] = t
        end
    end
    @inbounds for tt in 1:npair
        a = nb.pair_a[tt]
        b = nb.pair_b[tt]
        ba = a * p
        bb = b * p
        for k in 1:p, l in 1:p
            t += 1
            rows[t] = ba + k
            cols[t] = bb + l
            pair_pos[k, l, tt] = t
        end
    end

    # Sort the fixed (row, col) pattern once; `invord` recovers, for each
    # triplet built above, its final position in H.nzval, so later refills
    # never need to re-sort or rebuild the sparse structure.
    ord = sortperm(1:ntot; by = i -> (cols[i], rows[i]))
    invord = invperm(ord)
    rowval = rows[ord]

    colptr = zeros(Int, n + 1)
    colptr[1] = 1
    @inbounds for c in cols
        colptr[c + 1] += 1
    end
    cumsum!(colptr, colptr)

    H = SparseArrays.SparseMatrixCSC(n, n, colptr, rowval, zeros(FT, ntot))
    return CholeskySolverCache{FT}(H, diag_pos, pair_pos, invord, nothing)
end

"""
    _refill_cholesky!(cache, Hdiag, nb, live)

Write the current `Hdiag`/`nb.B` values into `cache.H.nzval` in place.  The
sparsity pattern built by [`_build_cholesky_cache`](@ref) never changes, so
this is the only per-outer-iteration cost; no triplets are rebuilt and no
sparse matrix is reallocated.

For any pair with a frozen (non-`live`) endpoint, `zero(FT)` is written
instead of `nb.B[:,:,tt]`.  This cannot be handled by structurally dropping
the pair (the whole point of the cached, fixed sparsity pattern is to reuse
CHOLMOD's symbolic factorization), but it does not need to be: with the
cross term zeroed and that star's right-hand side already `0` (frozen stars
are skipped by [`apply_JT!`](@ref)), the frozen star's row of
`(H + λI) δ = rhs` collapses to `Hdiag[frozen] · δ_frozen = 0`, and since
`Hdiag[frozen] + λI` is positive definite this gives `δ_frozen = 0` exactly
-- the same reduced system [`_pcg!`](@ref) solves via its `Ap` masking.
"""
function _refill_cholesky!(cache::CholeskySolverCache{FT}, Hdiag, nb::NeighborBlocks, live) where {FT}
    p, _, n_active = size(Hdiag)
    npair = length(nb)
    @inbounds for a in 1:n_active
        for k in 1:p, l in 1:k
            t = cache.diag_pos[k, l, a]
            cache.H.nzval[cache.invord[t]] = Hdiag[k, l, a]
        end
    end
    @inbounds for tt in 1:npair
        a = nb.pair_a[tt] + 1
        b = nb.pair_b[tt] + 1
        frozen_pair = !live[a] || !live[b]
        for k in 1:p, l in 1:p
            t = cache.pair_pos[k, l, tt]
            cache.H.nzval[cache.invord[t]] = frozen_pair ? zero(FT) : nb.B[k, l, tt]
        end
    end
    return cache.H
end

"""
    _solve_cholesky!(δ, cache, λ, rhs)

Solve `(H + λI) δ = rhs` using the cached sparse `H` (values already
refreshed by [`_refill_cholesky!`](@ref)).  The first call performs a full
symbolic + numeric factorization; every subsequent call reuses the symbolic
analysis via `cholesky!`, redoing only the numeric factorization for the
current `λ` shift, per §5.2/§6.2.
"""
function _solve_cholesky!(δ, cache::CholeskySolverCache, λ, rhs)
    Hs = Symmetric(cache.H, :L)
    if cache.F === nothing
        cache.F = cholesky(Hs; shift = λ)
    else
        cholesky!(cache.F, Hs; shift = λ)
    end
    δ .= cache.F \ rhs
    return δ
end

# ==============================================================================
# Position step cap (§6.3)
# ==============================================================================

function _cap_position_step!(δ, max_step, p, k_y, k_x)
    FT = eltype(δ)
    max_step = FT(max_step)
    (k_y === nothing && k_x === nothing) && return δ
    n_active = div(length(δ), p)
    @inbounds for a in 0:(n_active - 1)
        base = a * p
        dy = k_y === nothing ? zero(FT) : δ[base + k_y]
        dx = k_x === nothing ? zero(FT) : δ[base + k_x]
        len = sqrt(dy * dy + dx * dx)
        len > max_step || continue
        scale = max_step / len
        k_y === nothing || (δ[base + k_y] *= scale)
        k_x === nothing || (δ[base + k_x] *= scale)
    end
    return δ
end

# ==============================================================================
# Per-star convergence and freezing
# ==============================================================================

"""
    _freeze_star!(θ, data_work, model_template, free_names_val, fixed, p,
                  dy_off, dx_off, anchor_y, anchor_x, pixels, j)

Permanently remove star `j`'s current model contribution from `data_work` by
evaluating it once over its own stamp footprint (`O(S^2)`, not a masked call
into [`_render_model!`](@ref), which would redo `O(n_active * S^2)` work to
extract one star) and subtracting the result. Called exactly once, at the
outer iteration a star freezes: from then on the star is skipped by every
per-star loop (`_fill_stamps!`, `apply_JT!`, `_render_model!`), and its light
is already accounted for in `data_work`, so it never needs to be rendered
again. `θ` for a frozen star simply stops changing thereafter, so the final
diagnostics render (against the untouched original `data`, over every star
unconditionally) reproduces exactly what was subtracted here.
"""
function _freeze_star!(
        θ, data_work, model_template, free_names_val, fixed, p,
        dy_off, dx_off, anchor_y, anchor_x, pixels, j
    )
    S2 = length(dy_off)
    base = (j - 1) * p
    m = PSF.model_from_vector(model_template, free_names_val, view(θ, base + 1:base + p), fixed)
    ay = anchor_y[j]
    ax = anchor_x[j]
    @inbounds for mi in 1:S2
        fi = pixels[mi, j]
        fi != 0 || continue
        data_work[fi] -= evaluate(m, ay + dy_off[mi], ax + dx_off[mi])
    end
    return data_work
end

"""
    _freeze_remaining!(live, frozen_at, star_converged, iter)

Mark every currently-live star converged and frozen at outer iteration
`iter`, without rendering or subtracting anything. Used when a *global*
convergence test (the existing whole-system `g_tol`/`x_tol`/`f_tol` checks)
fires: the outer loop is about to exit, and the final diagnostics render
(§8 of `fit_all_stars_simultaneous`) unconditionally re-renders every star
at its final `θ` against the untouched original `data` regardless of `live`,
so no per-star subtraction is needed here.
"""
function _freeze_remaining!(live, frozen_at, star_converged, iter)
    @inbounds for j in eachindex(live)
        if live[j]
            live[j] = false
            frozen_at[j] = iter
            star_converged[j] = true
        end
    end
    return live
end

"""
    _compact_neighbors(nb::NeighborBlocks{FT}, live, p) where {FT}

Drop every pair with a non-`live` (frozen) endpoint from `nb`, returning a
new, smaller `NeighborBlocks`. A single linear scan of the existing flat
arrays -- no sort, unlike [`_build_neighbors`](@ref), since compaction only
ever removes entries, never regroups them. Star indices in the surviving
pairs are *not* renumbered. `B`'s values are not carried over: the next
`_accumulate_H!` call refills them from scratch regardless.

Purely a performance optimization (shrinks the per-pair loops that dominate
`_accumulate_H!`/`_apply_H!` at high crowding, per `experiments/simul-solve/
report.md` §6): correctness for the `:cg` solver does not depend on when, or
whether, this runs, because [`_pcg!`](@ref) already masks frozen-star slices
out of every `Ap` (see its docstring). Only called for `solver === :cg`; the
`:cholesky` path instead zeroes frozen-touching pair values in place (see
[`_refill_cholesky!`](@ref)) since its sparse pattern is fixed for the whole
fit.
"""
function _compact_neighbors(nb::NeighborBlocks{FT}, live, p) where {FT}
    npair = length(nb)
    nkeep = 0
    @inbounds for t in 1:npair
        (live[nb.pair_a[t] + 1] && live[nb.pair_b[t] + 1]) && (nkeep += 1)
    end
    nkeep == npair && return nb
    if nkeep == 0
        return NeighborBlocks{FT}(Int32[], Int32[], Int[], Int32[], Int32[], zeros(FT, p, p, 0))
    end
    new_pair_a = Vector{Int32}(undef, nkeep)
    new_pair_b = Vector{Int32}(undef, nkeep)
    new_offsets = Vector{Int}(undef, nkeep + 1)
    new_offsets[1] = 1
    total_new = 0
    @inbounds for t in 1:npair
        (live[nb.pair_a[t] + 1] && live[nb.pair_b[t] + 1]) || continue
        total_new += nb.offsets[t + 1] - nb.offsets[t]
    end
    new_shared_ma = Vector{Int32}(undef, total_new)
    new_shared_mb = Vector{Int32}(undef, total_new)
    bi = 0
    si = 0
    @inbounds for t in 1:npair
        (live[nb.pair_a[t] + 1] && live[nb.pair_b[t] + 1]) || continue
        bi += 1
        new_pair_a[bi] = nb.pair_a[t]
        new_pair_b[bi] = nb.pair_b[t]
        lo = nb.offsets[t]
        hi = nb.offsets[t + 1] - 1
        len = hi - lo + 1
        copyto!(new_shared_ma, si + 1, nb.shared_ma, lo, len)
        copyto!(new_shared_mb, si + 1, nb.shared_mb, lo, len)
        si += len
        new_offsets[bi + 1] = si + 1
    end
    return NeighborBlocks{FT}(
        new_pair_a, new_pair_b, new_offsets, new_shared_ma, new_shared_mb,
        zeros(FT, p, p, nkeep),
    )
end

"""
    _g_tol_local(j, p, colnorm_flat, b_scaled, C_r, g_tol)

Per-star analogue of the global gradient-cosine test in the outer loop
(`gnorm`): the same `max_k |b_true_k| / (sqrt(A_kk * C_r) + eps)` expression,
restricted to star `j`'s own `p`-parameter slice and reusing the same
shared, global `C_r = cost / dof` (`fitting_sparse_plan.md` §4.4 "Deviation
2": `C_r`'s denominator is deliberately global, not something to recompute
per star).
"""
function _g_tol_local(j, p, colnorm_flat, b_scaled, C_r, g_tol)
    FT = eltype(b_scaled)
    g_tiny = eps(FT)
    base = (j - 1) * p
    gnorm = zero(FT)
    @inbounds for k in 1:p
        i = base + k
        b_true = colnorm_flat[i] * b_scaled[i]
        A_ii = colnorm_flat[i]^2
        gnorm = max(gnorm, abs(b_true) / (sqrt(A_ii * C_r) + g_tiny))
    end
    return gnorm <= FT(g_tol)
end

"""
    _x_tol_local(j, p, D, colnorm_flat, δ_scaled, θ, x_tol)

Per-star analogue of the global rescaled-step-norm test in the outer loop
(`x_converged`), restricted to star `j`'s own `p`-parameter slice of the same
shared `D`/`colnorm_flat` arrays.
"""
function _x_tol_local(j, p, D, colnorm_flat, δ_scaled, θ, x_tol)
    FT = eltype(θ)
    base = (j - 1) * p
    num = zero(FT)
    xnorm = zero(FT)
    @inbounds for k in 1:p
        i = base + k
        v = (D[i] / colnorm_flat[i]) * δ_scaled[i]
        num += v * v
        xv = D[i] * θ[i]
        xnorm += xv * xv
    end
    return sqrt(num) <= FT(x_tol) * (sqrt(xnorm) + FT(x_tol))
end

# ==============================================================================
# Public entry point
# ==============================================================================

"""
    fit_all_stars_simultaneous(image, psf, sources, fit_rad; kws...) -> MultiPassPhotResult

Simultaneous PSF-fitting photometry: optimize the parameters of every source
at once with a damped Gauss-Newton / Levenberg-Marquardt loop over the normal
matrix `H = J'WJ`, accumulated directly from per-star stamp derivatives.  The
default solver is preconditioned conjugate gradient on `H`.

Every non-fitting concern (background, detection, morphology, PSF model,
diagnostics) is shared with [`fit_all_stars`](@ref); only the optimizer
differs.  The two functions return the same [`MultiPassPhotResult`](@ref),
with one field whose *unit* differs (see `# Returns` below).

Stars are frozen out of the fit independently, as soon as each one satisfies
a per-star convergence test: its model is subtracted out of the working
image once and it is never re-rendered or re-linearized again, so later
outer iterations only pay for stars still moving.  This has no effect on the
converged answer for an isolated star.  For a star in a genuinely blended
group, the freeze test only looks at that star's own step size/gradient
given its neighbors' *current* (not yet converged) positions -- there is no
gate on neighbor convergence, matching how DAOPHOT's ALLSTAR (the closest
precedent for a whole-frame simultaneous fitter) drops converged stars.  In
a strongly degenerate flux-sharing blend this can let a star freeze slightly
before the pair is jointly optimal, since it can no longer respond to
further motion from a still-active neighbor afterward -- a bias in the point
estimate itself, not just in the (already-approximate, see below) reported
error.

# Comparability contract

- Only `(y, x, flux)` may be free per star.  Fix `bkg` (subtract the
  background first and pass `fixed = (; bkg = zero(eltype(image)))`) and fix
  all shape parameters.  Any other free parameter raises an `ArgumentError`.
- The solver uses a fixed-size stamp of half-width `fit_rad` rounded up
  to the nearest integer.
  Diagnostics use the identical box, so `qfit`, `chisq`, and `crowding` are computed
  over the same pixels in both arms.
- No robust reweighting: `reweight`, `scale_estimator`, and `weight_reset_tol`
  are not accepted.


# Keyword arguments

- `inner_iterations::Int = 5`: CG iteration upper cap per damping trial.
- `cg_tol::Real = 1.0e-6`: relative-residual stopping tolerance for the
  inner CG solve, `‖rhs − Aδ‖ / ‖rhs‖ <= cg_tol`; lets an isolated or
  lightly-blended star's system (solved exactly by the block-Jacobi
  preconditioner) exit CG in 1-2 iterations instead of always burning the
  full cap. This is a safe performance optimization only: raising
  `inner_iterations` or tightening `cg_tol` to make the solve more exact is
  **not** a safe way to improve bright-star precision in a crowded field --
  it resolves genuinely near-degenerate, tightly-blended groups' weakly
  determined directions more precisely, which amplifies noise into large
  (sometimes non-positive) flux swings rather than converging to a better
  answer, with no outer-loop safeguard against it (`λ`/acceptance are driven
  by the global cost, not per-group). See `_pcg!`'s docstring for the full
  analysis.
- `max_step::Real = 0.25`: per-star position step cap in pixels. Too large
  risks shooting past the actual minimum in the direction of the gradient
  in a single step; too small risks slowing convergence.  Recommend setting
  this to a small factor of the initial centroid uncertainty so that a star
  can reach its true position in a few steps, but not catastrophically
  overshoot it.
- `solver::Symbol = :cg`: `:cg` (block-Jacobi PCG on `H`) or `:cholesky`
  (exact CHOLMOD oracle).
- `max_trials::Int = 8`: damping retries per outer iteration.
- `freeze_compaction_frac::Real = 0.05`: `solver = :cg` only.  Once at least
  this fraction of `n_active` stars have frozen since the last compaction,
  `NeighborBlocks` is compacted to drop pairs touching frozen stars.  A pure
  performance tuning knob (like `inner_iterations`/`max_trials`), not a
  correctness one: `:cg` masks frozen stars out of every CG iteration
  regardless of compaction cadence (see `_pcg!`'s docstring).
- `max_iter::Integer = 40` maximum number of outer iterations
- `x_tol::Real = 1.0e-5`: per-star rescaled step-norm convergence test
- `f_tol::Real = 1.0e-4`: whole-system relative cost improvement test
- `g_tol::Real = 1.0e-4`: per-star gradient-cosine convergence test
- `λ_init::Real = 1.0e-4`: initial damping factor
- `λ_up::Real = 10.0`: damping increase factor on a failed trial
- `λ_down::Real = 10.0`: damping decrease factor on a successful trial
- `λ_min::Real = 1.0e-12`: minimum damping factor
- `λ_max::Real = 1.0e12`: maximum damping factor
- `damping`: [`MarquardtDamping`](@ref) or [`AdaptiveDamping`](@ref) instance
- `show_trace::Bool = false`: print a trace of the fitting process
- `covariance_estimator`: a [`CovarianceEstimator`](@ref) instance to compute the covariance of the final fit.  If `nothing`, no covariance is computed.

# Returns

A [`MultiPassPhotResult`](@ref).  `converged[i]` is `true` if star `i` froze
via its own `x_tol`/`g_tol` test before `max_iter` iterations or the global
`f_tol` tolerance was reached; `false` otherwise.  `n_iter[i]` is the outer
iteration at which star `i` froze (or the final iteration count, for a star
never frozen). `n_passes` is the number of outer iterations actually run for
the fit as a whole. `n_failed`/`failure_msgs` report stars excluded before the solve.
"""
function fit_all_stars_simultaneous(
        image::AbstractMatrix{T},
        psf::AbstractPSFModel,
        sources,
        fit_rad::Real;
        fixed::NamedTuple = (;),
        inv_var = nothing,
        inner_iterations::Int = 5,
        cg_tol::Real = 1.0e-6,
        max_step::Real = 0.25,
        solver::Symbol = :cg,
        max_trials::Int = 8,
        freeze_compaction_frac::Real = 0.05,
        max_iter::Integer = 40,
        x_tol::Real = 1.0e-5,
        f_tol::Real = 1.0e-4,
        g_tol::Real = 1.0e-4,
        λ_init::Real = 1.0e-4,
        λ_up::Real = 10.0,
        λ_down::Real = 10.0,
        λ_min::Real = 1.0e-12,
        λ_max::Real = 1.0e12,
        damping = MarquardtDamping(),
        show_trace::Bool = false,
        covariance_estimator = nothing,
    ) where {T}
    FT = float(T)
    max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
    inner_iterations > 0 || throw(ArgumentError("inner_iterations must be positive"))
    cg_tol > 0 || throw(ArgumentError("cg_tol must be positive"))
    max_trials > 0 || throw(ArgumentError("max_trials must be positive"))
    0 < freeze_compaction_frac <= 1 || throw(ArgumentError("freeze_compaction_frac must be in (0, 1]"))
    solver in (:cg, :cholesky) || throw(ArgumentError("solver must be :cg or :cholesky, got $(repr(solver))"))
    damping isa LevenbergDamping && throw(ArgumentError(
        "LevenbergDamping is not supported by fit_all_stars_simultaneous: it " *
        "conflicts with the column-equilibrated x_tol scaling. Use MarquardtDamping."
    ))

    # Same default-selection rule as `lm_irls` (levenberg_marquardt.jl); there is
    # no IRLS reweighting here, so the choice reduces to whether `inv_var` was given.
    covariance_estimator = if !isnothing(covariance_estimator)
        covariance_estimator
    elseif !isnothing(inv_var)
        KnownWeightsCovarianceEstimator()
    else
        ReweightedCovarianceEstimator()
    end

    # -------------------------------------------------------------------
    # 1. Catalog, free-parameter metadata
    # -------------------------------------------------------------------
    params, errors = _extract_source_catalog(sources, psf, FT)
    n_params, n_stars = size(params)
    n_stars == 0 && return MultiPassPhotResult(
        FT[], FT[], FT[], FT[], FT[], FT[], FT[], FT[],
        falses(0), falses(0), FT[], FT[], FT[], FT[], FT[], Int[], Int(0), Int(0), String[], Matrix{FT}(undef, 0, 0),
    )

    prop_names = collect(keys(ConstructionBase.getproperties(psf)))
    @assert length(prop_names) == n_params

    free_names, free_idx, _ = PSF.free_params(psf, fixed)
    isempty(free_idx) && throw(ArgumentError("all model parameters are fixed; nothing to fit"))
    offenders = setdiff(free_names, (:y, :x, :flux))
    isempty(offenders) || throw(ArgumentError(
        "fit_all_stars_simultaneous fits only (y, x, flux) per star; got free " *
        "parameters $(offenders). Fix `bkg` (subtract the background first and " *
        "pass `fixed = (; bkg = zero(eltype(image)))`), and fix all shape " *
        "parameters. Pass the same `fixed` to `fit_all_stars` for a " *
        "like-for-like comparison."
    ))
    p = length(free_idx)

    row_y = findfirst(==(:y), prop_names)
    row_x = findfirst(==(:x), prop_names)
    row_flux = findfirst(==(:flux), prop_names)
    row_bkg = findfirst(==(:bkg), prop_names)
    row_y === nothing && throw(ArgumentError("PSF model has no `y` field"))
    row_x === nothing && throw(ArgumentError("PSF model has no `x` field"))
    row_flux === nothing && throw(ArgumentError("PSF model has no `flux` field"))

    # grad_col[k] = 1 (dy), 2 (dx), or 3 (dflux) for free param k.
    grad_col = [free_names[k] === :y ? 1 : (free_names[k] === :x ? 2 : 3) for k in 1:p]
    k_y = findfirst(==(1), grad_col)
    k_x = findfirst(==(2), grad_col)

    free_names_val = Val(free_names)
    is_fixed = Tuple(setdiff(1:n_params, free_idx))
    for k in is_fixed
        errors[k, :] .= zero(FT)
    end

    # -------------------------------------------------------------------
    # 2. Data and weights
    # -------------------------------------------------------------------
    ny, nx = size(image)
    npix = ny * nx
    data = vec(Matrix{FT}(image))
    if inv_var === nothing
        all(isfinite, data) || throw(ArgumentError(
            "the image contains non-finite values but `inv_var` was not provided; " *
            "pass `inv_var` to mask them or clean the image first"
        ))
        w = ones(FT, npix)
    else
        size(inv_var) == size(image) || throw(ArgumentError("`inv_var` must be the same size as `image`"))
        w = Vector{FT}(undef, npix)
        iv = inv_var
        @inbounds for idx in eachindex(image)
            wv = iv[idx]
            w[idx] = if isfinite(wv) && wv > 0
                FT(wv)
            else
                zero(FT)
            end
        end
    end
    # Non-finite data pixels are zeroed AND zero-weighted
    @inbounds for q in 1:npix
        if !isfinite(data[q])
            data[q] = zero(FT)
            w[q] = zero(FT)
        end
    end

    # -------------------------------------------------------------------
    # 3. Stamp footprint and neighbor structure (built once)
    # -------------------------------------------------------------------
    built = _build_stamps!(image, params, row_y, row_x, row_flux, fit_rad, w, FT)
    pixels = built.pixels
    anchor_y = built.anchor_y
    anchor_x = built.anchor_x
    dy_off = built.dy_off
    dx_off = built.dx_off
    active = built.active
    valid = built.valid
    n_failed = built.n_failed
    failure_msgs = built.failure_msgs
    S2 = built.S2
    n_active = length(active)

    n_active == 0 && return _empty_simultaneous_result(
        n_stars, params, errors, row_y, row_x, row_flux, row_bkg, valid, failure_msgs, FT, ny, nx)

    union_pix, nbr_blocks = _build_neighbors(pixels, S2, n_active, p, FT)
    # `nbr_blocks` is reassigned in place to a compacted (smaller) structure
    # as stars freeze when `solver === :cg` (step 6 below); `nbr_blocks_full`
    # keeps the original, un-compacted pair list -- covering every star pair
    # whose stamps ever overlapped, frozen or not -- for the end-of-fit
    # marginal-covariance computation in step 7, which needs every
    # cross-term regardless of freeze order.
    nbr_blocks_full = nbr_blocks
    chol_cache = solver === :cholesky ? _build_cholesky_cache(nbr_blocks, n_active, p) : nothing

    # -------------------------------------------------------------------
    # 4. Working buffers
    # -------------------------------------------------------------------
    n = p * n_active
    θ = Vector{FT}(undef, n)
    for (j, i) in enumerate(active)
        for k in 1:p
            θ[(j - 1) * p + k] = FT(params[free_idx[k], i])
        end
    end

    stamp = StampDerivatives{FT, Int32}(
        zeros(FT, p, S2, n_active), pixels, zeros(FT, p, n_active), npix, p, S2,
    )
    model_img = zeros(FT, npix)
    model_cand = zeros(FT, npix)
    wt_resid = zeros(FT, npix)
    render_buf = Vector{FT}(undef, S2)
    render_scratch = _render_scratch(psf, round(Int, sqrt(S2)), FT)
    fill_scratch = _fill_scratch(psf, round(Int, sqrt(S2)), FT)

    Hdiag = zeros(FT, p, p, n_active)
    Mblocks = zeros(FT, p, p, n_active)

    b_scaled = Vector{FT}(undef, n)
    δ_scaled = Vector{FT}(undef, n)
    δ = Vector{FT}(undef, n)
    rhs = Vector{FT}(undef, n)
    D = Vector{FT}(undef, n)
    r = Vector{FT}(undef, n)
    z = Vector{FT}(undef, n)
    pk = Vector{FT}(undef, n)
    Ap = Vector{FT}(undef, n)

    colnorm_flat = reshape(stamp.colnorm, n)

    # Per-star freezing state (§ "Per-star convergence and freezing" above).
    live = trues(n_active)
    frozen_at = zeros(Int, n_active)
    star_converged = falses(n_active)
    frozen_list = Int[]
    n_since_compaction = 0
    compaction_batch = max(1, ceil(Int, freeze_compaction_frac * n_active))
    data_work = copy(data)

    # -------------------------------------------------------------------
    # 5. Initial fill and cost
    # -------------------------------------------------------------------
    _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w, model_img,
        grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, fill_scratch)
    cost = _residual_cost!(wt_resid, model_img, data_work, w, union_pix)
    D .= colnorm_flat

    dof = max(length(union_pix) - n, 1)
    λ = FT(λ_init)
    converged = false
    n_run = 0

    # -------------------------------------------------------------------
    # 6. Outer damped Gauss-Newton / LM loop
    # -------------------------------------------------------------------
    for iter in 1:max_iter
        _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w, model_img,
            grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, fill_scratch)
        cost = _residual_cost!(wt_resid, model_img, data_work, w, union_pix)
        _accumulate_H!(Hdiag, nbr_blocks, stamp)
        solver === :cholesky && _refill_cholesky!(chol_cache, Hdiag, nbr_blocks, live)
        apply_JT!(b_scaled, stamp, wt_resid, live)

        # g_converged (levenberg_marquardt.jl:535-540), reduced-cost denominator.
        # Global fallback: frozen stars' b_scaled is exactly 0 (apply_JT! skips
        # them), so they never dominate this max; a pass here means the whole
        # remaining live set has stopped moving.
        C_r = cost / dof
        g_tiny = eps(FT)
        gnorm = zero(FT)
        @inbounds for i in 1:n
            b_true = colnorm_flat[i] * b_scaled[i]
            A_ii = colnorm_flat[i]^2
            gnorm = max(gnorm, abs(b_true) / (sqrt(A_ii * C_r) + g_tiny))
        end
        if gnorm <= g_tol
            show_trace && println("g_tol convergence triggered")
            _freeze_remaining!(live, frozen_at, star_converged, iter - 1)
            converged = true
            n_run = iter - 1
            break
        end

        accepted = false
        f_converged = false
        for trial in 1:max_trials
            @. rhs = -b_scaled
            if solver === :cg
                _build_precond!(Mblocks, Hdiag, λ)
                _pcg!(δ_scaled, Hdiag, nbr_blocks, λ, rhs, Mblocks, inner_iterations, r, z, pk, Ap, frozen_list, p, cg_tol)
            else
                _solve_cholesky!(δ_scaled, chol_cache, λ, rhs)
            end

            @. δ = δ_scaled / colnorm_flat
            _cap_position_step!(δ, max_step, p, k_y, k_x)
            @. δ_scaled = colnorm_flat * δ

            _render_model!(model_cand, psf, free_names_val, fixed, θ .+ δ, p,
                dy_off, dx_off, anchor_y, anchor_x, pixels, live, render_buf, render_scratch)
            cost_cand = _cost!(model_cand, data_work, w, union_pix)
            Δcost = cost - cost_cand

            # f_converged: realized relative cost improvement
            f_converged = (cost_cand < cost) && (Δcost / cost <= FT(f_tol))

            if show_trace
                status = cost_cand < cost ? "accepted" : "rejected"
                println("Iter $(lpad(iter, 4)) trial $(lpad(trial, 2)) | converged = $(@sprintf("%5.2f%%", round(100 * length(frozen_list) / length(live), RoundDown; digits=2))) | cost = $(@sprintf("%.4e", cost)) -> $(@sprintf("%.4e", cost_cand)) | λ = $(@sprintf("%.2e", λ)) | ||g|| = $(@sprintf("%8.4f", gnorm)) | $status")
                if trial == max_trials
                    println("*** Max inner trials reached (`inner_iterations`) during outer iteration $(iter) ***")
                end
            end

            if cost_cand < cost
                accepted = true
                θ .+= δ
                cost = cost_cand
                λ = max(λ / FT(λ_down), FT(λ_min))
                @. D = max(D, colnorm_flat)
                break
            else
                λ = min(λ * FT(λ_up), FT(λ_max))
            end
        end

        n_run = iter
        if f_converged
            show_trace && println("f_tol convergence triggered")
            _freeze_remaining!(live, frozen_at, star_converged, iter)
            converged = true
            break
        end
        if accepted
            # Per-star freeze test, only meaningful for a step that was
            # actually applied: evaluated only here, not on a rejected trial.
            for j in 1:n_active
                live[j] || continue
                if _x_tol_local(j, p, D, colnorm_flat, δ_scaled, θ, x_tol) ||
                        _g_tol_local(j, p, colnorm_flat, b_scaled, C_r, g_tol)
                    _freeze_star!(θ, data_work, psf, free_names_val, fixed, p,
                        dy_off, dx_off, anchor_y, anchor_x, pixels, j)
                    live[j] = false
                    frozen_at[j] = iter
                    star_converged[j] = true
                    push!(frozen_list, j - 1)
                    n_since_compaction += 1
                end
            end
            if solver === :cg && n_since_compaction >= compaction_batch
                nbr_blocks = _compact_neighbors(nbr_blocks, live, p)
                n_since_compaction = 0
            end
            if !any(live)
                converged = true
                break
            end
        end
    end

    # -------------------------------------------------------------------
    # 7. Final parameter bookkeeping and validity gate
    # -------------------------------------------------------------------
    for (j, i) in enumerate(active)
        for k in 1:p
            params[free_idx[k], i] = θ[(j - 1) * p + k]
        end
    end
    # Fixed parameters take the `fixed` value (like fit_all_stars).
    for fn in keys(fixed)
        row = findfirst(==(fn), prop_names)
        row === nothing && continue
        params[row, :] .= FT(getfield(fixed, fn))
        errors[row, :] .= zero(FT)
    end

    # Errors: rebuild stamp values at final θ (every star, live or frozen),
    # then invert the global H via Takahashi's recursion so marginal errors
    # include coupling with blended neighbors (KnownWeightsCovarianceEstimator
    # assumed; ReweightedCovarianceEstimator not yet supported here).
    _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w, model_img,
        grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, trues(n_active), fill_scratch)
    _accumulate_H!(Hdiag, nbr_blocks_full, stamp)
    err_cache = _build_cholesky_cache(nbr_blocks_full, n_active, p)
    _refill_cholesky!(err_cache, Hdiag, nbr_blocks_full, trues(n_active))

    shifts = FT(1.0e-12) .* FT(10) .^ (0:6) # escalate shift on PosDefException, as elsewhere in this file
    Σ_scaled = nothing
    for (attempt, shift) in enumerate(shifts)
        try
            println("Attempting covariance computation with shift = $shift")
            Σ_scaled = selected_inverse_diagonal_blocks(err_cache.H, p; shift)
            break
        catch e
            e isa PosDefException || rethrow()
            attempt == length(shifts) && rethrow()
        end
    end

    for (j, i) in enumerate(active)
        cov = zeros(FT, p, p)
        for k in 1:p, l in 1:p
            cov[k, l] = Σ_scaled[k, l, j] / (stamp.colnorm[k, j] * stamp.colnorm[l, j])
        end
        _extract_errors!(errors, cov, free_idx, is_fixed, i)
    end

    # Final validity gate: non-finite or non-positive flux after convergence.
    for i in 1:n_stars
        if valid[i] && !(isfinite(params[row_y, i]) && isfinite(params[row_x, i]) &&
                isfinite(params[row_flux, i]) && params[row_flux, i] > 0)
            valid[i] = false
        end
    end

    # -------------------------------------------------------------------
    # 8. Diagnostics and residual image
    # -------------------------------------------------------------------
    # Every star, live or frozen, against the untouched original `data` (not
    # data_work): a frozen star's θ has simply stopped changing since it froze.
    _render_model!(model_img, psf, free_names_val, fixed, θ, p,
        dy_off, dx_off, anchor_y, anchor_x, pixels, trues(n_active), render_buf, render_scratch)

    chisq = zeros(FT, n_stars)
    qfit = fill(convert(FT, NaN), n_stars)
    qfit_expected = fill(convert(FT, NaN), n_stars)
    qfit_z = fill(convert(FT, NaN), n_stars)
    crowding = fill(convert(FT, NaN), n_stars)

    global_resid = data .- model_img
    resid_mat = reshape(global_resid, ny, nx)
    # Reuse model_cand (no longer needed) as the neighbor-subtracted residual
    # buffer: clean_resid = global_residual + own_model.
    clean_resid = reshape(model_cand, ny, nx)

    for (j, i) in enumerate(active)
        valid[i] || continue
        m = PSF.model_from_vector(psf, free_names_val, view(θ, (j - 1) * p + 1:j * p), fixed)
        FT_fit = FT(fit_rad)
        yr = floor(Int, m.y - FT_fit):ceil(Int, m.y + FT_fit)
        xr = floor(Int, m.x - FT_fit):ceil(Int, m.x + FT_fit)
        yr, xr = _clamp_inds(yr, xr, image)
        (isempty(yr) || isempty(xr)) && continue
        inds = CartesianIndices((yr, xr))
        for pix in inds
            clean_resid[pix] = resid_mat[pix] + evaluate(m, pix)
        end
        _star_diagnostics!(qfit, qfit_expected, qfit_z, crowding, i, m,
            image, clean_resid, inds, inv_var, p)
        num = zero(FT)
        n_pix_good = 0
        for pix in inds
            iv = inv_var !== nothing ? inv_var[pix] : one(FT)
            if isfinite(iv) && iv > 0
                num += iv * resid_mat[pix]^2
                n_pix_good += 1
            end
        end
        den = n_pix_good - p
        if den > 0
            chisq[i] = num / den
        end
    end

    residual = reshape(global_resid, ny, nx)

    # -------------------------------------------------------------------
    # 9. Assemble result
    # -------------------------------------------------------------------
    y = params[row_y, :]
    x = params[row_x, :]
    flux = params[row_flux, :]
    bkg = row_bkg === nothing ? zeros(FT, n_stars) : params[row_bkg, :]

    y_err = errors[row_y, :]
    x_err = errors[row_x, :]
    flux_err = errors[row_flux, :]
    bkg_err = row_bkg === nothing ? zeros(FT, n_stars) : errors[row_bkg, :]

    # Per-star converged/n_iter: star_converged/frozen_at are indexed by the
    # stable 1:n_active index; map back to catalog index i = active[j], the
    # same pattern already used above for θ/errors. A star still live when
    # the loop exited (max_iter reached) is not converged; its n_iter is the
    # final outer-iteration count n_run.
    converged = falses(n_stars)
    n_iter = zeros(Int, n_stars)
    for (j, i) in enumerate(active)
        converged[i] = star_converged[j]
        n_iter[i] = live[j] ? n_run : frozen_at[j]
    end

    return MultiPassPhotResult(
        y, x, y_err, x_err, flux, flux_err, bkg, bkg_err,
        converged, valid, chisq, qfit, qfit_expected, qfit_z, crowding,
        n_iter, n_run, n_failed, failure_msgs, residual,
    )
end

function _empty_simultaneous_result(n_stars, params, errors, row_y, row_x, row_flux, row_bkg, valid, failure_msgs, FT, ny, nx)
    bkg = row_bkg === nothing ? zeros(FT, n_stars) : params[row_bkg, :]
    bkg_err = row_bkg === nothing ? zeros(FT, n_stars) : errors[row_bkg, :]
    return MultiPassPhotResult(
        params[row_y, :], params[row_x, :], errors[row_y, :], errors[row_x, :],
        params[row_flux, :], errors[row_flux, :], bkg, bkg_err,
        falses(n_stars), valid, zeros(FT, n_stars),
        fill(convert(FT, NaN), n_stars), fill(convert(FT, NaN), n_stars),
        fill(convert(FT, NaN), n_stars), fill(convert(FT, NaN), n_stars),
        zeros(Int, n_stars), Int(0), n_stars, failure_msgs, zeros(FT, ny, nx),
    )
end
