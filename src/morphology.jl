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

The returned moments are **not** centralised — the caller must compute
the centroid offset `μ_y = M10 / M00`, `μ_x = M01 / M00` and subtract
to obtain central moments.

# Returns
`(; M00, M10, M01, M20, M02, M11)` where each
```math
M_{pq} = \\sum_{y,x} w_{y,x} \\; \\max(0, z_{y,x}) \\; (y - y_0)^p \\; (x - x_0)^q
```
with ``w = \\mathtt{inv\\_var}`` and ``z = \\mathtt{image} - \\mathtt{background}``.
Pixels with ``w \\le 0`` are skipped.  If ``M_{00} \\le 0`` (all pixels
below background or fully masked), `M00 = 0` and higher moments are
meaningless; the caller should guard against this.
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
    bg = FT(background)
    fy0 = FT(y0)
    fx0 = FT(x0)
    @inbounds for idx in CartesianIndices(image)
        w = inv_var[idx]
        w > 0 || continue
        z = FT(image[idx]) - bg
        z > 0 || continue
        dy = FT(idx[1]) - fy0
        dx = FT(idx[2]) - fx0
        wz = w * z
        M00 += wz
        M10 += wz * dy
        M01 += wz * dx
        M20 += wz * dy * dy
        M02 += wz * dx * dx
        M11 += wz * dx * dy
    end
    return (; M00, M10, M01, M20, M02, M11)
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
  usually sufficient; the function computes the precise centre-of-mass
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
`(; fwhm, roundness1_aperture, roundness2_aperture, flux, centroid)` where

- `fwhm::NamedTuple (; y, x, theta)`: moment-based full width at half
  maximum along the ``y`` (row) and ``x`` (column) axes, and the
  position angle `theta` of the major axis in degrees CCW from the
  ``x``-axis (column direction).  ``\theta = 0`` means the major axis
  is aligned with columns.
- `roundness1_aperture::T`: DAOPHOT SROUND / photutils `roundness1`
  over the full cutout: ``2\cdot\Sigma_2/\Sigma_4`` where
  ``\Sigma_2`` is the weighted bilateral asymmetry and ``\Sigma_4``
  the weighted fourfold normalisation, summed over all valid pixels
  relative to the measured centroid.
  0 = symmetric, nonzero = asymmetric.
- `roundness2_aperture::T`: DAOPHOT GROUND convention from the full
  cutout: ``2(\sqrt{\sigma^2_{yy}} - \sqrt{\sigma^2_{xx}})/(\sqrt{\sigma^2_{yy}} + \sqrt{\sigma^2_{xx}})``.
  0 = circular, negative = extended in x (columns),
  positive = extended in y (rows).
- `flux::T`: total weighted flux ``M_{00}`` above background.
- `centroid::NamedTuple (; y, x)`: centre-of-mass centroid computed
  from the weighted moments.

If ``M_{00} \le 0`` (all pixels at or below background), every field
is `NaN`.  If ``\sigma^2_{yy} \le 0`` or ``\sigma^2_{xx} \le 0``
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

julia> abs(result.roundness2_aperture) < 1e-10 # nearly circular
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
                 flux = FT_M00,
                 centroid = (; y = n, x = n))
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

    # roundness1_aperture: full-aperture SROUND (DAOPHOT / photutils
    # roundness1 — bilateral vs. fourfold symmetry over all pixels).
    # Quadrants are defined relative to the measured centroid.
    # Non-positive inv_var or below-background pixels are skipped.
    sum2 = zero(FT)
    sum4 = zero(FT)
    @inbounds for idx in CartesianIndices(image)
        w = inv_var[idx]
        w > 0 || continue
        z = FT(image[idx]) - FT(background)
        # Skip the central pixel (infinitesimally close to centroid).
        dy = FT(idx[1]) - fwhm_cen
        dx = FT(idx[2]) - fwhm_cen_x
        dy == 0 && dx == 0 && continue
        wz = w * z
        # Quadrant signs from photutils: -quad1 + quad2 - quad3 + quad4.
        if dy >= 0 && dx > 0       # quad1: bottom-right
            sum2 -= wz
        elseif dy > 0 && dx <= 0   # quad2: bottom-left
            sum2 += wz
        elseif dy <= 0 && dx < 0   # quad3: top-left
            sum2 -= wz
        else                        # quad4: top-right (dy < 0, dx >= 0)
            sum2 += wz
        end
        sum4 += abs(wz)
    end
    roundness1_aperture = if sum4 > eps(FT)
        2 * sum2 / sum4
    else
        zero(FT)
    end

    # roundness2_aperture in the DAOPHOT GROUND convention:
    # HX ∝ 1/√σ²_xx, HY ∝ 1/√σ²_yy, so
    # GROUND = 2·(HX-HY)/(HX+HY) = 2·(√σ²_yy - √σ²_xx)/(√σ²_yy + √σ²_xx).
    # 0 = circular, negative = extended in x, positive = extended in y.
    sqrt_σyy = sqrt(σ²_yy)
    sqrt_σxx = sqrt(σ²_xx)
    denom_a = sqrt_σyy + sqrt_σxx
    roundness2_aperture = if denom_a > eps(FT)
        2 * (sqrt_σyy - sqrt_σxx) / denom_a
    else
        zero(FT)
    end

    return (; fwhm = (; y = fwhm_y, x = fwhm_x, theta),
             roundness1_aperture, roundness2_aperture,
             flux = FT_M00,
             centroid = (; y = fwhm_cen, x = fwhm_cen_x))
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
