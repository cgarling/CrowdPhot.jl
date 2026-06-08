module Background

import Random
using Statistics: mean, median, std

export AbstractBackgroundEstimator, AbstractBackgroundRMSEstimator
export MeanBackground, MedianBackground, SExtractorBackground,
       MMMBackground, BiweightLocationBackground
export StdRMS, MADStdRMS, BiweightScaleRMS
export Background2D, estimate_background

###############################################################################
# Abstract types

"""
    AbstractBackgroundEstimator

Supertype for all scalar background location estimators.

Concrete subtypes must implement `(::MyEstimator)(data::AbstractVector)`,
returning a scalar background estimate for the pixel values in `data`.
"""
abstract type AbstractBackgroundEstimator end

"""
    AbstractBackgroundRMSEstimator

Supertype for all background RMS (scatter) estimators.

Concrete subtypes must implement `(::MyEstimator)(data::AbstractVector)`,
returning a scalar RMS estimate for the pixel values in `data`.
"""
abstract type AbstractBackgroundRMSEstimator end

###############################################################################
# Internal utilities

# Median absolute deviation (unnormalized)
@inline function _mad(data::AbstractVector)
    m = median(data)
    return median(abs.(data .- m))
end

# Iterative sigma clipping: return a new vector containing only the
# unclipped values.  The result is always a fresh copy.
function _sigma_clip(data::AbstractVector, σ_low::Real, σ_high::Real = σ_low;
                     maxiters::Integer = 10)
    d = collect(float.(data))
    for _ in 1:maxiters
        isempty(d) && break
        m = median(d)
        s = std(d; corrected = false)
        s == 0 && break
        lo, hi = m - σ_low * s, m + σ_high * s
        d_new = filter(x -> lo ≤ x ≤ hi, d)
        length(d_new) == length(d) && break
        d = d_new
    end
    return d
end

# Biweight location (Tukey 1977; Beers, Flynn & Gebhardt 1990).
function _biweight_location(data::AbstractVector, c::Real = 6.0)
    T  = float(eltype(data))
    M  = median(data)
    S  = _mad(data)
    S == 0 && return T(M)
    u  = @. (data - M) / (c * S)
    w  = [abs(ui) < 1 ? (1 - ui^2)^2 : zero(T) for ui in u]
    sw = sum(w)
    sw == 0 && return T(M)
    return T(M) + sum((d - M) * wi for (d, wi) in zip(data, w)) / sw
end

# Biweight scale (square-root of biweight midvariance; Beers et al. 1990).
function _biweight_scale(data::AbstractVector, c::Real = 9.0)
    T   = float(eltype(data))
    n   = length(data)
    n == 0 && return zero(T)
    M   = median(data)
    S   = _mad(data)
    S == 0 && return zero(T)
    u   = @. (data - M) / (c * S)
    g   = @. abs(u) < 1
    num = sum((data[i] - M)^2 * (1 - u[i]^2)^4 for i in eachindex(data) if g[i]; init = zero(T))
    den = abs(sum((1 - 5 * u[i]^2) for i in eachindex(data) if g[i]; init = zero(T)))^2
    den == 0 && return zero(T)
    return sqrt(n * num / den)
end

# Collect valid (unmasked, finite) pixel values as a flat float vector.
function _valid_pixels(image::AbstractMatrix, mask::Nothing)
    return [float(v) for v in image if isfinite(v)]
end

function _valid_pixels(image::AbstractMatrix, mask::AbstractMatrix{Bool})
    size(image) == size(mask) ||
        throw(DimensionMismatch("image and mask must have the same size"))
    return [float(image[i]) for i in eachindex(image) if !mask[i] && isfinite(image[i])]
end

###############################################################################
# Location estimators

"""
    MeanBackground()

Estimate the background as the unweighted arithmetic mean of the pixel values.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> MeanBackground()(fill(3.0, 10))
3.0
```
"""
struct MeanBackground <: AbstractBackgroundEstimator end
(::MeanBackground)(data::AbstractVector) = mean(data)

"""
    MedianBackground()

Estimate the background as the sample median of the pixel values.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> MedianBackground()(fill(3.0, 10))
3.0
```
"""
struct MedianBackground <: AbstractBackgroundEstimator end
(::MedianBackground)(data::AbstractVector) = median(data)

"""
    SExtractorBackground()

Background estimator modelled on the mode statistic used in SExtractor
(Bertin & Arnouts 1996).

When the pixel distribution is roughly symmetric the background is estimated
as `2.5 × median − 1.5 × mean`.  If the distribution is strongly skewed
(`|mean − median| / σ > 0.3`) the median is returned instead, and when the
standard deviation is zero the mean is returned.  This rule makes the estimator
more resistant to source contamination than the mean while being roughly 30 %
noisier than the clipped mean in clean sky regions.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> SExtractorBackground()(fill(5.0, 20))
5.0

julia> SExtractorBackground()([5.0, 5.1, 4.9, 5.0, 5.2, 4.8])
5.0
```
"""
struct SExtractorBackground <: AbstractBackgroundEstimator end

function (::SExtractorBackground)(data::AbstractVector)
    m   = mean(data)
    med = median(data)
    s   = std(data; corrected = false)
    s ≈ 0                         && return m
    abs(m - med) / s > 0.3        && return med
    return 2.5 * med - 1.5 * m
end

"""
    MMMBackground(; median_factor=3, mean_factor=2)

Background estimator based on the MMM algorithm from DAOPHOT
(Stetson 1987).

Returns `median_factor × median − mean_factor × mean`.  With the default
coefficients this equals `3 × median − 2 × mean`.  The estimator assumes that
source contamination introduces a positive skew to the pixel distribution, so
it down-weights the mean relative to the median.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> MMMBackground()(fill(7.0, 15))
7.0

julia> MMMBackground(; median_factor=4, mean_factor=3)(fill(7.0, 15))
7.0
```
"""
Base.@kwdef struct MMMBackground <: AbstractBackgroundEstimator
    median_factor::Float64 = 3.0
    mean_factor::Float64   = 2.0
end

(alg::MMMBackground)(data::AbstractVector) =
    alg.median_factor * median(data) - alg.mean_factor * mean(data)

"""
    BiweightLocationBackground(; c=6.0)

Estimate the background using the robust biweight location statistic
(Tukey 1977; Beers, Flynn & Gebhardt 1990).

The tuning constant `c` controls the rejection threshold: pixels whose
standardised residual (normalised by the median absolute deviation)
exceeds `c` receive zero weight.  `c = 6` is the conventional default.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> BiweightLocationBackground()(fill(4.0, 20))
4.0

julia> BiweightLocationBackground(; c=4.0)(fill(4.0, 20))
4.0
```
"""
Base.@kwdef struct BiweightLocationBackground <: AbstractBackgroundEstimator
    c::Float64 = 6.0
end

(alg::BiweightLocationBackground)(data::AbstractVector) =
    _biweight_location(float.(data), alg.c)

###############################################################################
# RMS estimators

"""
    StdRMS()

Estimate the background RMS as the (population) standard deviation.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> StdRMS()(fill(1.0, 10))
0.0
```
"""
struct StdRMS <: AbstractBackgroundRMSEstimator end
(::StdRMS)(data::AbstractVector) = std(data; corrected = false)

"""
    MADStdRMS()

Estimate the background RMS via the normalised median absolute deviation:

```math
\\hat{\\sigma} \\approx 1.4826 \\times \\mathrm{MAD}
```

This is a robust scale estimate that is less sensitive to outliers
than the standard deviation.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> MADStdRMS()(fill(1.0, 10))
0.0
```
"""
struct MADStdRMS <: AbstractBackgroundRMSEstimator end
(::MADStdRMS)(data::AbstractVector) = 1.4826 * _mad(data)

"""
    BiweightScaleRMS(; c=9.0)

Estimate the background RMS as the biweight scale (square-root of the
biweight midvariance; Beers, Flynn & Gebhardt 1990).

The tuning constant `c = 9` is the standard default.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> BiweightScaleRMS()(fill(2.0, 10))
0.0
```
"""
Base.@kwdef struct BiweightScaleRMS <: AbstractBackgroundRMSEstimator
    c::Float64 = 9.0
end

(alg::BiweightScaleRMS)(data::AbstractVector) =
    _biweight_scale(float.(data), alg.c)

###############################################################################
# Scalar estimation

"""
    estimate_background(image;
        estimator        = SExtractorBackground(),
        rms_estimator    = StdRMS(),
        mask             = nothing,
        sigma            = 3.0,
        maxiters         = 10) -> NamedTuple{(:bkg, :bkg_rms)}

Estimate a global (scalar) background and background RMS for `image`.

Non-finite values are always excluded.  Where `mask` is provided, pixels
at `true` positions are excluded before any other processing.  After
masking, iterative sigma clipping is applied when `sigma` is not `nothing`.

Returns a `NamedTuple` `(bkg = …, bkg_rms = …)`.

# Arguments
- `estimator`: an [`AbstractBackgroundEstimator`](@ref) or any callable
  `f(v::AbstractVector) -> scalar`.
- `rms_estimator`: an [`AbstractBackgroundRMSEstimator`](@ref) or any callable.
- `mask`: `AbstractMatrix{Bool}` with the same shape as `image`; `true`
  excludes the pixel.  `nothing` (default) uses all finite pixels.
- `sigma`: sigma-clipping threshold.  Set to `nothing` to disable clipping.
- `maxiters`: maximum number of sigma-clipping iterations.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> img = fill(100.0, 64, 64);

julia> r = estimate_background(img)
(bkg = 100.0, bkg_rms = 0.0)

julia> r = estimate_background(img; estimator=MMMBackground(), rms_estimator=MADStdRMS())
(bkg = 100.0, bkg_rms = 0.0)
```
"""
function estimate_background(
        image::AbstractMatrix{<:Real};
        estimator::Union{AbstractBackgroundEstimator, Function} = SExtractorBackground(),
        rms_estimator::Union{AbstractBackgroundRMSEstimator, Function} = StdRMS(),
        mask::Union{Nothing, AbstractMatrix{Bool}} = nothing,
        sigma::Union{Nothing, Real} = 3.0,
        maxiters::Integer = 10,
    )
    pixels = _valid_pixels(image, mask)
    isempty(pixels) && throw(ArgumentError("no finite pixels available for background estimation"))
    if !isnothing(sigma)
        pixels = _sigma_clip(pixels, sigma; maxiters)
        isempty(pixels) && throw(ArgumentError("all pixels were removed by sigma clipping"))
    end
    return (bkg = estimator(pixels), bkg_rms = rms_estimator(pixels))
end

###############################################################################
# Bicubic Catmull-Rom zoom interpolator (no external dependencies)

@inline function _catmull_rom(p0::T, p1::T, p2::T, p3::T, t::T) where {T <: AbstractFloat}
    return 0.5 * ((2 * p1) +
                  (-p0 + p2) * t +
                  (2 * p0 - 5 * p1 + 4 * p2 - p3) * t^2 +
                  (-p0 + 3 * p1 - 3 * p2 + p3) * t^3)
end

# Map output pixel k (1-indexed) in a dimension of length K_out to the
# corresponding mesh coordinate (1-indexed) in a mesh of length K_mesh.
@inline function _zoom_coord(k::Int, K_out::Int, K_mesh::Int)
    K_mesh == 1 && return 1.0
    K_out  == 1 && return 1.0
    return 1.0 + (k - 1) * (K_mesh - 1) / (K_out - 1)
end

@inline _clamp_idx(i::Int, n::Int) = max(1, min(n, i))

"""
    _bicubic_zoom(mesh, H, W)

Upsample `mesh` (size `M × N`) to a `H × W` array using bicubic
Catmull-Rom interpolation.  Corner samples are preserved exactly.
"""
function _bicubic_zoom(mesh::AbstractMatrix{T}, H::Integer, W::Integer) where {T <: Real}
    M, N  = size(mesh)
    F     = float(T)
    out   = Matrix{F}(undef, H, W)
    fmesh = F === T ? mesh : F.(mesh)
    for j in 1:W
        yf   = _zoom_coord(j, W, N)
        jm   = floor(Int, yf)
        ty   = F(yf - jm)
        jm1  = _clamp_idx(jm - 1, N)
        jm2  = _clamp_idx(jm,     N)
        jm3  = _clamp_idx(jm + 1, N)
        jm4  = _clamp_idx(jm + 2, N)
        for i in 1:H
            xf   = _zoom_coord(i, H, M)
            im_  = floor(Int, xf)
            tx   = F(xf - im_)
            r1   = _clamp_idx(im_ - 1, M)
            r2   = _clamp_idx(im_,     M)
            r3   = _clamp_idx(im_ + 1, M)
            r4   = _clamp_idx(im_ + 2, M)
            # Interpolate along the column axis for each of the four rows.
            q1 = _catmull_rom(fmesh[r1, jm1], fmesh[r1, jm2], fmesh[r1, jm3], fmesh[r1, jm4], ty)
            q2 = _catmull_rom(fmesh[r2, jm1], fmesh[r2, jm2], fmesh[r2, jm3], fmesh[r2, jm4], ty)
            q3 = _catmull_rom(fmesh[r3, jm1], fmesh[r3, jm2], fmesh[r3, jm3], fmesh[r3, jm4], ty)
            q4 = _catmull_rom(fmesh[r4, jm1], fmesh[r4, jm2], fmesh[r4, jm3], fmesh[r4, jm4], ty)
            out[i, j] = _catmull_rom(q1, q2, q3, q4, tx)
        end
    end
    return out
end

###############################################################################
# 2D median filter (no ImageFiltering dependency)

# In-place 2-D median filter with boundary replication.
function _median_filter2d!(dst::Matrix{T}, src::Matrix{T},
                            fh::Int, fw::Int) where {T <: AbstractFloat}
    M, N = size(src)
    hh, hw = fh ÷ 2, fw ÷ 2
    buf = Vector{T}(undef, fh * fw)
    for i in 1:M, j in 1:N
        k = 0
        for di in -hh:hh, dj in -hw:hw
            k += 1
            buf[k] = src[_clamp_idx(i + di, M), _clamp_idx(j + dj, N)]
        end
        dst[i, j] = median(view(buf, 1:k))
    end
    return dst
end

function _median_filter2d(arr::Matrix{T}, fh::Int, fw::Int) where {T <: AbstractFloat}
    (fh == 1 && fw == 1) && return copy(arr)
    return _median_filter2d!(similar(arr), arr, fh, fw)
end

###############################################################################
# Mesh NaN filling by iterative nearest-neighbour propagation

function _fill_nans!(mesh::Matrix{T}) where {T <: AbstractFloat}
    any(isnan, mesh) || return mesh
    M, N    = size(mesh)
    changed = true
    while changed
        changed = false
        for i in 1:M, j in 1:N
            isnan(mesh[i, j]) || continue
            s, k = zero(T), 0
            for di in -1:1, dj in -1:1
                (di == 0 && dj == 0) && continue
                v = mesh[_clamp_idx(i + di, M), _clamp_idx(j + dj, N)]
                if !isnan(v)
                    s += v; k += 1
                end
            end
            if k > 0
                mesh[i, j] = s / k
                changed     = true
            end
        end
    end
    # Fallback: fill any remaining NaNs with the global mean of finite values.
    if any(isnan, mesh)
        vals = [v for v in mesh if !isnan(v)]
        gm = isempty(vals) ? zero(T) : sum(vals) / length(vals)
        for i in eachindex(mesh)
            isnan(mesh[i]) && (mesh[i] = gm)
        end
    end
    return mesh
end

###############################################################################
# 2D background estimation

"""
    Background2D

Result of spatially varying background estimation over a mesh grid.

Fields
------
- `background`      — full-resolution background map (same size as the input image).
- `background_rms`  — full-resolution background-RMS map.
- `mesh_background` — low-resolution per-box background estimates.
- `mesh_background_rms` — low-resolution per-box background-RMS estimates.
- `box_size`        — the `(rows, cols)` box dimensions used.

See the constructor [`Background2D(image, box_size; …)`](@ref) for details.
"""
struct Background2D{T <: AbstractFloat}
    background::Matrix{T}
    background_rms::Matrix{T}
    mesh_background::Matrix{T}
    mesh_background_rms::Matrix{T}
    box_size::Tuple{Int, Int}
end

"""
    Background2D(image, box_size;
        estimator          = SExtractorBackground(),
        rms_estimator      = StdRMS(),
        mask               = nothing,
        filter_size        = (3, 3),
        exclude_percentile = 10.0,
        sigma              = 3.0,
        maxiters           = 10,
        edge_method        = :pad) -> Background2D

Estimate a spatially varying background by tiling `image` into boxes,
estimating the background (and RMS) in each box, optionally median-filtering
the mesh, and upsampling the result back to the original image size via a
bicubic Catmull-Rom interpolator.

`box_size` may be an integer (square boxes) or a `(rows, cols)` tuple.
The image is padded with `NaN` along the right/bottom edges when it is not
an exact multiple of `box_size`; pass `edge_method = :crop` to crop instead.

A box is excluded from the mesh (and filled by interpolation from neighbours)
when the fraction of valid (unmasked, finite, not sigma-clipped) pixels falls
below `exclude_percentile / 100 * box_npixels`.

Non-finite image values are automatically masked; an error is thrown if the
result contains more non-finite values than covered by `mask`.

# Arguments
- `estimator`: background location estimator (default [`SExtractorBackground`](@ref)).
- `rms_estimator`: background RMS estimator (default [`StdRMS`](@ref)).
- `mask`: `AbstractMatrix{Bool}` where `true` marks pixels to exclude.
- `filter_size`: window size of the 2-D median filter applied to the mesh
  (must be odd; default `(3, 3)`; use `1` or `(1,1)` to skip filtering).
- `exclude_percentile`: boxes with fewer than this percent of valid pixels
  are excluded and filled by interpolation.
- `sigma`: sigma-clipping threshold (`nothing` disables clipping).
- `maxiters`: maximum sigma-clipping iterations.
- `edge_method`: `:pad` (default) or `:crop`.

# Examples
```jldoctest
julia> using CrowdPhot.Background

julia> img = fill(200.0, 128, 128);

julia> b = Background2D(img, 32);

julia> b.background ≈ fill(200.0, 128, 128)
true

julia> all(iszero, b.background_rms)
true
```
"""
function Background2D(
        image::AbstractMatrix{<:Real},
        box_size::Union{Integer, NTuple{2, <:Integer}} = 64;
        estimator::Union{AbstractBackgroundEstimator, Function} = SExtractorBackground(),
        rms_estimator::Union{AbstractBackgroundRMSEstimator, Function} = StdRMS(),
        mask::Union{Nothing, AbstractMatrix{Bool}} = nothing,
        filter_size::Union{Integer, NTuple{2, <:Integer}} = (3, 3),
        exclude_percentile::Real = 10.0,
        sigma::Union{Nothing, Real} = 3.0,
        maxiters::Integer = 10,
        edge_method::Symbol = :pad,
    )
    H, W    = size(image)
    bh, bw  = _to_pair(box_size)
    fh, fw  = _to_pair(filter_size)
    (isone(fh) || isodd(fh)) && (isone(fw) || isodd(fw)) ||
        throw(ArgumentError("filter_size must be odd; got ($fh, $fw)"))
    0 ≤ exclude_percentile ≤ 100 ||
        throw(ArgumentError("exclude_percentile must be in [0, 100]"))

    # Validate mask / non-finite check.
    if !isnothing(mask)
        size(mask) == (H, W) ||
            throw(DimensionMismatch("mask must be the same size as image"))
    end
    # Warn about non-finite values not covered by the mask.
    for i in eachindex(image)
        if !isfinite(image[i]) && (isnothing(mask) || !mask[i])
            @warn "image contains non-finite values that are not covered by the mask; they will be excluded automatically"
            break
        end
    end

    # Tile the image with padding or cropping.
    padded, nmesh_rows, nmesh_cols = _tile_image(Val(edge_method), image, bh, bw)
    pad_H, pad_W = size(padded)

    # Build a combined boolean mask (true = excluded).
    total_mask = _build_total_mask(padded, mask, H, W, pad_H, pad_W)

    T = float(eltype(image))
    mesh_bkg = fill(T(NaN), nmesh_rows, nmesh_cols)
    mesh_rms = fill(T(NaN), nmesh_rows, nmesh_cols)

    min_valid = exclude_percentile / 100.0 * bh * bw

    for mi in 1:nmesh_rows, mj in 1:nmesh_cols
        rows = ((mi - 1) * bh + 1):(mi * bh)
        cols = ((mj - 1) * bw + 1):(mj * bw)
        view_img  = view(padded, rows, cols)
        view_mask = view(total_mask, rows, cols)
        pixels    = [float(view_img[k]) for k in eachindex(view_img)
                     if !view_mask[k] && isfinite(view_img[k])]
        length(pixels) < min_valid && continue
        if !isnothing(sigma)
            pixels = _sigma_clip(pixels, sigma; maxiters)
            isempty(pixels) && continue
        end
        mesh_bkg[mi, mj] = estimator(pixels)
        mesh_rms[mi, mj] = rms_estimator(pixels)
    end

    all(isnan, mesh_bkg) &&
        throw(ArgumentError("all mesh boxes were excluded; try larger box_size, " *
                            "smaller exclude_percentile, or looser sigma clipping"))

    # Fill excluded (NaN) mesh cells by iterative nearest-neighbour propagation.
    _fill_nans!(mesh_bkg)
    _fill_nans!(mesh_rms)

    # Optional 2-D median filter on the mesh.
    fmesh_bkg = _median_filter2d(mesh_bkg, fh, fw)
    fmesh_rms = _median_filter2d(mesh_rms, fh, fw)

    # Upsample to the padded image size and crop to the original size.
    # For :pad, output is cropped back to (H, W).  For :crop, the output
    # is smaller — nmesh * box_size ≤ (H, W) — so no extra cropping is needed.
    out_H = min(H, pad_H)
    out_W = min(W, pad_W)
    full_bkg = _bicubic_zoom(fmesh_bkg, pad_H, pad_W)[1:out_H, 1:out_W]
    full_rms = _bicubic_zoom(fmesh_rms, pad_H, pad_W)[1:out_H, 1:out_W]

    return Background2D{T}(full_bkg, full_rms, mesh_bkg, mesh_rms, (bh, bw))
end

###############################################################################
# Internal helpers for Background2D

_to_pair(x::Integer) = (Int(x), Int(x))
_to_pair(x::NTuple{2, <:Integer}) = (Int(x[1]), Int(x[2]))

function _tile_image(::Val{:pad}, image, bh, bw)
    H, W  = size(image)
    nextra_r = H % bh
    nextra_c = W % bw
    pad_r = nextra_r == 0 ? 0 : bh - nextra_r
    pad_c = nextra_c == 0 ? 0 : bw - nextra_c
    pad_H = H + pad_r
    pad_W = W + pad_c
    T     = float(eltype(image))
    padded = fill(T(NaN), pad_H, pad_W)
    padded[1:H, 1:W] .= image
    nmesh_rows = pad_H ÷ bh
    nmesh_cols = pad_W ÷ bw
    return padded, nmesh_rows, nmesh_cols
end

function _tile_image(::Val{:crop}, image, bh, bw)
    H, W  = size(image)
    nmesh_rows = H ÷ bh
    nmesh_cols = W ÷ bw
    T     = float(eltype(image))
    cropped = T.(image[1:nmesh_rows * bh, 1:nmesh_cols * bw])
    return cropped, nmesh_rows, nmesh_cols
end

function _build_total_mask(padded, mask::Nothing, H, W, pad_H, pad_W)
    total = falses(pad_H, pad_W)
    # Padded region is excluded (the padded values are NaN, picked up automatically).
    total[(H + 1):pad_H, :] .= true
    total[:, (W + 1):pad_W] .= true
    return total
end

function _build_total_mask(padded, mask::AbstractMatrix{Bool}, H, W, pad_H, pad_W)
    total = falses(pad_H, pad_W)
    total[1:H, 1:W] .= mask
    total[(H + 1):pad_H, :] .= true
    total[:, (W + 1):pad_W] .= true
    return total
end

end # module Background