# morphology.jl — morphological measurements for stellar image cutouts.
#
# Provides `measure_star_shape` for aperture-based FWHM, roundness, and
# position angle from inverse-variance-weighted second moments.  The
# 3×3 core sharpness and roundness are available from `centroid_poly`
# (see centroids.jl).

# ---------------------------------------------------------------------------
# Internal: raw weighted second moments
# ---------------------------------------------------------------------------

"""
    _moments2(image, inv_var, background, y0, x0) -> NamedTuple

Compute raw (non-central) inverse-variance-weighted second moments of
`max(0, image .- background)` about the reference point `(y0, x0)`.
Mask invalid image pixels by setting their inverse variance to zero.

The returned moments are **not** centralised — the caller must compute
the centroid offset `μ_y = M10 / M00`, `μ_x = M01 / M00` and subtract
to obtain central moments.

# Returns
`(; M00, M10, M01, M20, M02, M11, W00, W10, W01, W20, W02, W11,
    sum2, sum4, aperture_sum, aperture_area, aperture_sum_err)`
where each flux moment is
```math
M_{pq} = \\sum_{y,x} w_{y,x} \\; \\max(0, z_{y,x}) \\; (y - y_0)^p \\; (x - x_0)^q
```
with ``w = \\mathtt{inv\\_var}`` and ``z = \\mathtt{image} - \\mathtt{background}``.
The ``W_{pq}`` fields are the corresponding weight-only moments
``\\sum w_{y,x}(y-y_0)^p(x-x_0)^q`` over the same included pixels, used
for delta-method centroid covariance propagation.
Pixels with ``w \\le 0`` are skipped.  If ``M_{00} \\le 0`` (all pixels
below background or fully masked), `M00 = 0` and higher moments are
meaningless; the caller should guard against this.

`sum2` and `sum4` are the raw SROUND accumulators (bilateral asymmetry
and fourfold normalization) computed over all valid pixels with
``w > 0``, using the reference point ``(y_0, x_0)`` as a proxy for the
centroid.  The caller must divide ``2\\cdot\\mathrm{sum2}/\\mathrm{sum4}``
to obtain the SROUND value.

`aperture_sum` is the unweighted rectangular-cutout sum of
``z = \\mathtt{image} - \\mathtt{background}`` over pixels with positive
inverse variance.  `aperture_area` is the number of pixels in that sum,
and `aperture_sum_err` is the formal propagated uncertainty
``\\sqrt{\\sum 1/w}`` assuming independent pixel errors.
"""
function _moments2(
        image::AbstractMatrix{T},
        inv_var::AbstractMatrix,
        background::Real,
        y0::Real,
        x0::Real,
    ) where {T}
    FT = float(T)
    M00 = zero(FT)
    M10 = zero(FT)
    M01 = zero(FT)
    M20 = zero(FT)
    M02 = zero(FT)
    M11 = zero(FT)
    # Weight-only moments for delta-method centroid covariance.
    # Delta method takes ~ 20% longer than using flux-weighted
    # second moments, but should be more accurate for faint sources
    # where the flux-weighted second moments can be noisy and even
    # negative. This is not a significant bottleneck for full-pipeline
    # runs so we can afford the extra computation.
    W00 = zero(FT)
    W10 = zero(FT)
    W01 = zero(FT)
    W20 = zero(FT)
    W02 = zero(FT)
    W11 = zero(FT)
    # SROUND accumulators (bilateral asymmetry sums).
    sum2 = zero(FT)
    sum4 = zero(FT)
    # Unweighted rectangular aperture diagnostics over valid pixels.
    aperture_sum = zero(FT)
    aperture_area = 0
    aperture_var = zero(FT)
    bg = FT(background)
    fy0 = FT(y0)
    fx0 = FT(x0)
    @inbounds for idx in CartesianIndices(image)
        w = inv_var[idx]
        w > 0 || continue
        fw = FT(w)
        z = FT(image[idx]) - bg

        # Aperture sums keep signed residuals over the same unmasked cutout.
        aperture_sum += z
        aperture_area += 1
        aperture_var += inv(fw)

        # From here, moments only use pixels above background.
        z > 0 || continue
        dy = FT(idx[1]) - fy0
        dx = FT(idx[2]) - fx0
        wz = fw * z
        M00 += wz
        M10 += wz * dy
        M01 += wz * dx
        M20 += wz * dy * dy
        M02 += wz * dx * dx
        M11 += wz * dx * dy
        W00 += fw
        W10 += fw * dy
        W01 += fw * dx
        W20 += fw * dy * dy
        W02 += fw * dx * dx
        W11 += fw * dx * dy
        # SROUND: exclude the central pixel (matching DAOPHOT/photutils).
        if !(dy == 0 && dx == 0)
            if dy <= 0 && dx > 0         # top-right + center-right (photutils quad1)
                sum2 -= wz
            elseif dy < 0 && dx <= 0     # top-left + top-center (photutils quad2)
                sum2 += wz
            elseif dy >= 0 && dx < 0     # bottom-left + center-left (photutils quad3)
                sum2 -= wz
            elseif dy > 0 && dx >= 0     # bottom-right + bottom-center (photutils quad4)
                sum2 += wz
            end
            sum4 += abs(wz)
        end
    end
    return (; M00, M10, M01, M20, M02, M11,
             W00, W10, W01, W20, W02, W11, sum2, sum4,
             aperture_sum, aperture_area, aperture_sum_err = sqrt(aperture_var))
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@doc raw"""
    measure_star_shape(image, y0, x0; inv_var, background, fwhm_factor) -> NamedTuple

Compute aperture-based morphological measurements for a stellar image
cutout using inverse-variance-weighted second central moments.

# Arguments
- `image::AbstractMatrix`: image cutout of a single star.
- `y0::Real, x0::Real`: approximate centroid around which raw moments
  are accumulated.  Integer pixel coordinates (e.g. the peak pixel) are
  usually sufficient; the function computes the precise center-of-mass
  from the moments themselves.
- `inv_var::AbstractMatrix`: per-pixel inverse variance, same size as
  `image`.  Defaults to `Fill(one(float(eltype(image))), size(image))`
  (uniform weighting).  Set entries to zero to mask bad pixels.
- `background::Real`: scalar background level subtracted before computing
  moments.  Defaults to `0`.  Pixels with ``\mathtt{image} - \mathtt{background} \le 0``
  are excluded from the moment sum.
- `fwhm_factor::Real`: scale factor from Gaussian σ to FWHM.
  Defaults to ``2\sqrt{2\log 2} \approx 2.35482``.

# Returns
`(; fwhm, roundness1_aperture, roundness2_aperture, moment_norm,
    aperture_sum, aperture_area, aperture_sum_err, centroid)` where

- `fwhm::NamedTuple (; y, x, theta)`: moment-based, axis-aligned marginal
  full width at half maximum along the ``y`` (row) and ``x`` (column)
  axes.  These are not principal-axis widths for a rotated source.
  `theta` is the position angle of the covariance major axis in degrees,
  measured counter-clockwise from the ``+x``-axis (column direction).
  ``\theta = 0`` means the major axis is aligned with columns;
  positive ``\theta`` rotates toward rows.
- `roundness1_aperture::T`: DAOPHOT SROUND / photutils `roundness1`
  over the full cutout: ``2\cdot\Sigma_2/\Sigma_4`` where
  ``\Sigma_2`` is the weighted bilateral asymmetry and ``\Sigma_4``
  the weighted fourfold normalization.
  0 = symmetric, nonzero = asymmetric.
- `roundness2_aperture::T`: DAOPHOT GROUND / photutils `roundness2`
  convention: ``2(\sqrt{\sigma^2_{yy}} - \sqrt{\sigma^2_{xx}})/(\sqrt{\sigma^2_{yy}} + \sqrt{\sigma^2_{xx}})``.
  0 = circular, negative = extended in x (columns),
  positive = extended in y (rows).
- `moment_norm::T`: weighted zeroth moment ``M_{00}`` used to normalize
  the shape moments.  When `inv_var` is not uniform this is not a physical
  source flux and should not be used for photometric calibration.
- `aperture_sum::T`: unweighted sum of `image - background` over unmasked
  pixels in the rectangular cutout.
- `aperture_area::Int`: number of pixels contributing to `aperture_sum`.
- `aperture_sum_err::T`: formal inverse-variance propagated uncertainty
  on `aperture_sum`, assuming independent pixel errors.
- `centroid::NamedTuple (; y, x, y_err, x_err, cov)`: center-of-mass
  centroid, 1-σ uncertainties, and 2×2 `SMatrix` covariance.

If ``M_{00} \le 0`` (all pixels at or below background), shape and
centroid fields are `NaN`; aperture-sum diagnostics are still reported.
If ``\sigma^2_{yy} \le 0`` or ``\sigma^2_{xx} \le 0``
(the distribution has no measurable width, e.g. a single bright pixel),
`fwhm.y` and `fwhm.x` are `NaN` and `roundness2_aperture` is `0`
(the denominator vanishes → degenerate, treated as isotropic).

# Examples
```jldoctest
julia> using CrowdPhot: measure_star_shape

julia> img = [0.1 0.3 0.1; 0.3 1.0 0.3; 0.1 0.3 0.1];

julia> result = measure_star_shape(img; background=0);

julia> result.fwhm.y ≈ result.fwhm.x ≈ 1.4603973964538084
true

julia> result.centroid.y ≈ 2.0
true

julia> result.centroid.x ≈ 2.0
true

julia> result.centroid.y_err > 0
true
```
"""
function measure_star_shape(
        image::AbstractMatrix{T},
        y0::Real,
        x0::Real;
        inv_var::AbstractMatrix = Fill(one(float(T)), size(image)),
        background::Real = zero(float(T)),
        fwhm_factor::Real = 2.3548200450309493,
    ) where {T}
    FT = float(T)

    mom = _moments2(image, inv_var, background, y0, x0)
    FT_M00 = FT(mom.M00)

    if FT_M00 <= zero(FT)
        n = FT(NaN)
        return (; fwhm = (; y = n, x = n, theta = n),
                 roundness1_aperture = n, roundness2_aperture = n,
                 moment_norm = FT_M00,
                 aperture_sum = FT(mom.aperture_sum),
                 aperture_area = mom.aperture_area,
                 aperture_sum_err = FT(mom.aperture_sum_err),
                 centroid = (; y = n, x = n, y_err = n, x_err = n,
                              cov = @SMatrix [n n; n n]))
    end

    inv_M00 = inv(FT_M00)
    μ_y = FT(mom.M10) * inv_M00
    μ_x = FT(mom.M01) * inv_M00

    # Centralised second moments: σ²_pq = M_pq / M00 - μ_p μ_q.
    σ²_yy = FT(mom.M20) * inv_M00 - μ_y * μ_y
    σ²_xx = FT(mom.M02) * inv_M00 - μ_x * μ_x
    σ²_xy = FT(mom.M11) * inv_M00 - μ_x * μ_y

    # Clamp negative variances (possible from noise on faint sources).
    σ²_yy = max(zero(FT), σ²_yy)
    σ²_xx = max(zero(FT), σ²_xx)

    ff = FT(fwhm_factor)
    fwhm_y = σ²_yy > 0 ? ff * sqrt(σ²_yy) : FT(NaN)
    fwhm_x = σ²_xx > 0 ? ff * sqrt(σ²_xx) : FT(NaN)

    # Position angle from the 2×2 covariance.
    # theta = 0.5 * atan(2σ²_xy, σ²_xx - σ²_yy), converted to degrees.
    num = 2 * σ²_xy
    den = σ²_xx - σ²_yy
    # For a nearly isotropic profile, both num and den are ~0 and the
    # angle is undefined.  Use a tolerance rather than exact zero because
    # floating-point summation order can leave σ²_yy ≠ σ²_xx at ~1e-16.
    if abs(num) + abs(den) <= eps(FT) * (σ²_yy + σ²_xx)
        theta = zero(FT)
    else
        theta = FT(rad2deg(atan(num, den) / 2))
    end

    fwhm_cen = FT(y0) + μ_y
    fwhm_cen_x = FT(x0) + μ_x

    # roundness1_aperture: SROUND from the sums accumulated in _moments2.
    # Uses the reference point (y0,x0) as a proxy for the centroid; the
    # error from the centroid offset is negligible for well-centered cutouts.
    roundness1_aperture = if mom.sum4 > eps(FT)
        2 * FT(mom.sum2) / FT(mom.sum4)
    else
        zero(FT)
    end

    # roundness2_aperture in the DAOPHOT GROUND / photutils roundness2
    # convention: 0 = circular, negative = extended in x, positive =
    # extended in y.
    sqrt_σyy = sqrt(σ²_yy)
    sqrt_σxx = sqrt(σ²_xx)
    denom_a = sqrt_σyy + sqrt_σxx
    roundness2_aperture = if denom_a > eps(FT)
        2 * (sqrt_σyy - sqrt_σxx) / denom_a
    else
        zero(FT)
    end

    # Centroid covariance from the delta method for the ratio estimator.
    inv_M00_sq = inv_M00 * inv_M00
    cent_cov_yy = (FT(mom.W20) - 2 * μ_y * FT(mom.W10) +
                   μ_y * μ_y * FT(mom.W00)) * inv_M00_sq
    cent_cov_xx = (FT(mom.W02) - 2 * μ_x * FT(mom.W01) +
                   μ_x * μ_x * FT(mom.W00)) * inv_M00_sq
    cent_cov_xy = (FT(mom.W11) - μ_y * FT(mom.W01) - μ_x * FT(mom.W10) +
                   μ_y * μ_x * FT(mom.W00)) * inv_M00_sq

    return (; fwhm = (; y = fwhm_y, x = fwhm_x, theta),
             roundness1_aperture, roundness2_aperture,
             moment_norm = FT_M00,
             aperture_sum = FT(mom.aperture_sum),
             aperture_area = mom.aperture_area,
             aperture_sum_err = FT(mom.aperture_sum_err),
             centroid = (; y = fwhm_cen, x = fwhm_cen_x,
                          y_err = sqrt(max(zero(FT), cent_cov_yy)),
                          x_err = sqrt(max(zero(FT), cent_cov_xx)),
                          cov = @SMatrix [cent_cov_yy cent_cov_xy;
                                          cent_cov_xy cent_cov_xx]))
end

"""
    measure_star_shape(image; kws...) -> NamedTuple

Convenience method that finds the brightest pixel in `image` via
`findmax` and calls [`measure_star_shape`](@ref measure_star_shape)
with those integer coordinates.  See the core method for keyword
arguments and return fields.
"""
function measure_star_shape(image::AbstractMatrix; kws...)
    _, maxidx = findmax(image)
    i0, j0 = Tuple(maxidx)
    return measure_star_shape(image, Int(i0), Int(j0); kws...)
end

# ---------------------------------------------------------------------------
# Batch measurement from matched-filter results
# ---------------------------------------------------------------------------

"""
    _default_half_width(result::MatchedFilterResult) -> Int

Return a sensible default half-width for cutout extraction around each
detected peak.  The kernel radius (half the kernel size) is used as a
proxy for the PSF extent, with a 2-pixel margin added to capture the
wings.  The minimum half-width is 3, guaranteeing at least a 7×7 cutout.
"""
function _default_half_width(result::MatchedFilterResult)
    kr = size(result.kernel, 1) ÷ 2
    kc = size(result.kernel, 2) ÷ 2
    return max(3, max(kr, kc) + 2)
end

"""
    measure_star_shapes(result::MatchedFilterResult; kws...) -> Vector{NamedTuple}

Measure centroid, shape, and morphological properties for every peak
detected by [`matched_filter`](@ref).

For each peak in `result.peaks`, this function:

1. Extracts a square cutout of size ``(2 \\times \\mathtt{half\\_width} + 1)^2``
   centered on the peak pixel from the original image.
2. Calls [`centroid_poly`](@ref) on the 3×3 core to obtain a sub-pixel
   centroid (polynomial and center-of-mass) and core diagnostics
   (normalized curvature, roundness).
3. Calls [`choose_centroid`](@ref) to select the best centroid estimate.
4. Calls [`measure_star_shape`](@ref) on the full cutout to compute
   aperture-based morphology and rectangular aperture sums.

All coordinate fields in the returned NamedTuples are in **global**
pixel coordinates of the original image.

# Arguments

- `result::MatchedFilterResult`: result from [`matched_filter`](@ref).
- `half_width::Int`: half-width of the square cutout extracted around
  each peak.  The cutout size is ``(2 \\times \\mathtt{half\\_width} + 1)
  \\times (2 \\times \\mathtt{half\\_width} + 1)`` pixels.  Defaults to the
  kernel radius plus 2, with a minimum of 3.
- `background::Real`: scalar background level subtracted before computing
  image moments.  Defaults to `0`.  Passed to [`measure_star_shape`](@ref).
- `fwhm_factor::Real`: scale factor from Gaussian σ to FWHM.  Defaults to
  ``2\\sqrt{2\\log 2} \\approx 2.35482``.  Passed to [`measure_star_shape`](@ref).
- `peaks::Union{AbstractVector{Int}, Nothing}`: optional vector of integer
  indices into `result.peaks` specifying which peaks to measure.  When
  `nothing` (default), all peaks are measured (subject to
  `min_significance`).
- `min_significance::Union{Real, Nothing}`: optional significance threshold;
  only peaks with `result.peak_significances[i] >= min_significance` are
  measured.  Ignored if `peaks` is also provided.

# Returns

A `Vector` of `NamedTuple`s, one per measured peak.  Each `NamedTuple`
has the following fields:

- `peak_index::Int`: index into the original `result.peaks` array.
- `pixel::CartesianIndex{2}`: the peak pixel `(row, column)` in the
  original image.
- `significance`: detection significance at this peak.
- `matched_filter_flux`: matched-filter flux estimate at this peak.
- `core`: the full [`centroid_poly`](@ref) result — `(; poly, com,
  normalized_curvature, roundness1_core, roundness2_core)` with
  coordinates in global pixels.
- `centroid`: the chosen centroid `(; y, x, source)` from
  [`choose_centroid`](@ref) in global pixels.  `source` is `:poly` or `:com`.
- `morphology`: the full [`measure_star_shape`](@ref) result — `(; fwhm,
  roundness1_aperture, roundness2_aperture, moment_norm, aperture_sum,
  aperture_area, aperture_sum_err, centroid)` with
  coordinates in global pixels.

!!! note
    If a peak is so close to the image border that no full 3×3
    neighbourhood exists, all fields in `core` are `NaN` and
    `centroid.source` is `:poly` (degenerate).  The `morphology` fields
    are computed from the available (clipped) cutout and may still be
    valid.

# Examples

```jldoctest
julia> using CrowdPhot: matched_filter, measure_star_shapes

julia> mf_result = matched_filter(zeros(50, 50), 3.0; sigma=0.0);

julia> results = measure_star_shapes(mf_result);

julia> isempty(results)
true
```

# References
See [Vakili2016](@citet) for the polynomial centroid method.
"""
function measure_star_shapes(
        result::MatchedFilterResult{T};
        half_width::Int = _default_half_width(result),
        background::Real = zero(T),
        fwhm_factor::Real = 2.3548200450309493,
        peaks::Union{AbstractVector{Int}, Nothing} = nothing,
        min_significance::Union{Real, Nothing} = nothing,
    ) where {T}
    # Determine which peaks to measure.
    all_peak_idx = if peaks !== nothing
        collect(Int, peaks)
    elseif min_significance !== nothing
        FT = float(T)
        findall(sig -> sig >= FT(min_significance), result.peak_significances)
    else
        collect(1:length(result.peaks))
    end

    n = length(all_peak_idx)
    n == 0 && return NamedTuple[]

    H, W = size(result.image)
    FT = float(T)
    return map(all_peak_idx) do pidx
        pixel = result.peaks[pidx]
        i0, j0 = Tuple(pixel)  # row, column

        # Extract cutout with boundary clipping.
        y_start = max(1, i0 - half_width)
        y_end   = min(H, i0 + half_width)
        x_start = max(1, j0 - half_width)
        x_end   = min(W, j0 + half_width)

        cutout = @view result.image[y_start:y_end, x_start:x_end]

        ivar_cutout = if result.inv_var !== nothing
            @view result.inv_var[y_start:y_end, x_start:x_end]
        else
            Fill(one(T), size(cutout))
        end

        # Peak pixel within the cutout (1-indexed).
        i0_cut = Int(i0 - y_start + 1)
        j0_cut = Int(j0 - x_start + 1)

        # Global offset for coordinate conversion.
        dy_global = FT(y_start - 1)
        dx_global = FT(x_start - 1)

        # 1. Polynomial centroid on the 3×3 core.
        core_local = centroid_poly(cutout, i0_cut, j0_cut, ivar_cutout)

        # Convert core coordinates to global.
        core_global = (;
            poly = (;
                y = core_local.poly.y + dy_global,
                x = core_local.poly.x + dx_global,
                peak = core_local.poly.peak,
                y_err = core_local.poly.y_err,
                x_err = core_local.poly.x_err,
                peak_err = core_local.poly.peak_err,
                cov = core_local.poly.cov,
            ),
            com = (;
                y = core_local.com.y + dy_global,
                x = core_local.com.x + dx_global,
                y_err = core_local.com.y_err,
                x_err = core_local.com.x_err,
                cov = core_local.com.cov,
            ),
            normalized_curvature = core_local.normalized_curvature,
            roundness1_core = core_local.roundness1_core,
            roundness2_core = core_local.roundness2_core,
        )

        # 2. Choose best centroid.
        chosen = choose_centroid(core_local)
        centroid_global = (;
            y = chosen.y + dy_global,
            x = chosen.x + dx_global,
            source = chosen.source,
        )

        # 3. Aperture morphology.
        morph_local = measure_star_shape(
            cutout, i0_cut, j0_cut;
            inv_var = ivar_cutout,
            background = background,
            fwhm_factor = fwhm_factor,
        )

        # Convert morphology centroid to global.
        morph_global = (;
            fwhm = morph_local.fwhm,
            roundness1_aperture = morph_local.roundness1_aperture,
            roundness2_aperture = morph_local.roundness2_aperture,
            moment_norm = morph_local.moment_norm,
            aperture_sum = morph_local.aperture_sum,
            aperture_area = morph_local.aperture_area,
            aperture_sum_err = morph_local.aperture_sum_err,
            centroid = (;
                y = morph_local.centroid.y + dy_global,
                x = morph_local.centroid.x + dx_global,
                y_err = morph_local.centroid.y_err,
                x_err = morph_local.centroid.x_err,
                cov = morph_local.centroid.cov,
            ),
        )

        return (;
            peak_index = pidx,
            pixel = pixel,
            significance = result.peak_significances[pidx],
            matched_filter_flux = result.peak_fluxes[pidx],
            core = core_global,
            centroid = centroid_global,
            morphology = morph_global,
        )
    end
end
