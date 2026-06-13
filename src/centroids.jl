# ---------------------------------------------------------------------------
# 3x3 polynomial centroiding  —  Vakili & Hogg (2016), arXiv:1610.05873
#
# The 3x3 patch is indexed in row-major order.  Local coordinates (y, x)
# follow the global convention: y is the row offset (first index), x is the
# column offset (second index), each in {-1, 0, 1} relative to the center pixel:
#
#   (-1,-1)  (-1, 0)  (-1, 1)
#   ( 0,-1)  ( 0, 0)  ( 0, 1)
#   ( 1,-1)  ( 1, 0)  ( 1, 1)
#
# Design-matrix row for pixel i: [1, x_i, y_i, x_i², x_i*y_i, y_i²]
# (the polynomial is P(x,y) = a + bx + cy + dx² + exy + fy² where x tracks
# column offset and y tracks row offset).
#
# The normal-equation matrix N = Aᵀ W A and right-hand side Aᵀ W z are built
# in closed form from weighted geometric moments S_pq = Σ w_i x_i^p y_i^q,
# exploiting that x,y ∈ {-1,0,1} collapses higher powers (x³=x, x⁴=x², …).
# ---------------------------------------------------------------------------

"""
    _centroid_poly3(image, inv_var) -> NamedTuple

Fit a quadratic 2-D polynomial ``P(x,y) = a + bx + cy + dx² + exy + fy²``
to a 3×3 patch using weighted least squares with inverse-variance weights
`inv_var`.

# Returns
A `NamedTuple` with keys `(; poly, com, normalized_curvature, roundness1_core,
roundness2_core)`:

- `poly`: `NamedTuple` `(; y, x, peak, y_err, x_err, peak_err, cov)` with
  the polynomial centroid (row, column) relative to the patch center, the
  fitted value at the centroid, 1-σ uncertainties, and 3×3 `SMatrix`
  covariance of `(y, x, peak)`.
- `com`: `NamedTuple` `(; y, x, y_err, x_err, cov)` with the
  inverse-variance-weighted center-of-mass centroid and its 2×2
  covariance on the same 3×3 patch.
- `normalized_curvature`: negated Laplacian divided by the fitted peak
  value, ``-(2d + 2f)/\\mathrm{peak} \\approx 2/\\mathrm{FWHM}^2`` for a
  Gaussian.  Flux-independent; ~0.5 for a typical stellar PSF, orders of
  magnitude larger for cosmic rays.
- `roundness1_core`: DAOPHOT SROUND / photutils `roundness1` convention:
  ``2\\cdot\\Sigma_2/\\Sigma_4`` — ratio of bilateral (2-fold) to fourfold
  symmetry of the 8 neighbour pixels.  0 = symmetric, nonzero = asymmetric.
- `roundness2_core`: DAOPHOT GROUND / photutils `roundness2` convention:
  ``2(\\sqrt{|d|} - \\sqrt{|f|})/(\\sqrt{|d|} + \\sqrt{|f|})``.
  0 = circular core, negative = extended in x (columns),
  positive = extended in y (rows).
  When the curvature matrix is near-singular the regularization biases
  this toward 0.

The design matrix is fixed (local coordinates `{-1,0,1}²`), so the
only free inputs are the 9 pixel values and 9 inverse-variance weights.

If the curvature matrix `D = [2d  e;  e  2f]` is near-singular, a small Tikhonov-style regularization is
added to its diagonal before computing the centroid.  If the data are
so noisy that the regularized determinant is still effectively zero,
the covariance will be large but the centroid estimates remain finite.


!!! note
    This function assumes the inputs are valid 3×3 matrices.  Border
    checking (whether a full 3×3 neighbourhood exists around the peak
    pixel) is the caller's responsibility — see [`centroid_poly`](@ref).
    `NaN` and `Inf` pixel values are not checked; they should be handled by
    a higher-level function.  Bad or saturated pixels can be masked by
    setting the corresponding entries in `inv_var` to zero.

# References
See [Vakili2016](@citet) for details.
"""
function _centroid_poly3(image::AbstractMatrix{T}, inv_var::AbstractMatrix{T}) where {T <: Real}
    # Coordinates:
    #
    #   image[1,1] image[1,2] image[1,3]     y = -1
    #   image[2,1] image[2,2] image[2,3]     y =  0
    #   image[3,1] image[3,2] image[3,3]     y =  1
    #
    #      x=-1       x=0       x=1
    #
    # Design row is (1, x, y, x^2, x*y, y^2).

    @inbounds begin
        z11 = image[1,1]; z12 = image[1,2]; z13 = image[1,3]
        z21 = image[2,1]; z22 = image[2,2]; z23 = image[2,3]
        z31 = image[3,1]; z32 = image[3,2]; z33 = image[3,3]

        w11 = inv_var[1,1]; w12 = inv_var[1,2]; w13 = inv_var[1,3]
        w21 = inv_var[2,1]; w22 = inv_var[2,2]; w23 = inv_var[2,3]
        w31 = inv_var[3,1]; w32 = inv_var[3,2]; w33 = inv_var[3,3]
    end

    wz11 = w11 * z11; wz12 = w12 * z12; wz13 = w13 * z13
    wz21 = w21 * z21; wz22 = w22 * z22; wz23 = w23 * z23
    wz31 = w31 * z31; wz32 = w32 * z32; wz33 = w33 * z33

    # Weighted geometric moments S_pq = Σ w_i x_i^p y_i^q.
    S00 = w11 + w12 + w13 + w21 + w22 + w23 + w31 + w32 + w33

    S10 = (w13 + w23 + w33) - (w11 + w21 + w31)
    S01 = (w31 + w32 + w33) - (w11 + w12 + w13)

    S20 = w11 + w13 + w21 + w23 + w31 + w33
    S02 = w11 + w12 + w13 + w31 + w32 + w33

    S11 = w11 - w13 - w31 + w33

    S21 = (w31 + w33) - (w11 + w13)
    S12 = (w13 + w33) - (w11 + w31)
    S22 = w11 + w13 + w31 + w33

    # Right-hand side r = A'Wz.
    R00 = wz11 + wz12 + wz13 + wz21 + wz22 + wz23 + wz31 + wz32 + wz33

    R10 = (wz13 + wz23 + wz33) - (wz11 + wz21 + wz31)
    R01 = (wz31 + wz32 + wz33) - (wz11 + wz12 + wz13)

    R20 = wz11 + wz13 + wz21 + wz23 + wz31 + wz33
    R02 = wz11 + wz12 + wz13 + wz31 + wz32 + wz33

    R11 = wz11 - wz13 - wz31 + wz33

    # Normal matrix: N = A'WA.  Aliases exploit x³=x, x⁴=x² (and same for y).
    Nmat = @SMatrix [
        S00 S10 S01 S20 S11 S02
        S10 S20 S11 S10 S21 S12
        S01 S11 S02 S21 S12 S01
        S20 S10 S21 S20 S11 S22
        S11 S21 S12 S11 S22 S11
        S02 S12 S01 S22 S11 S02
    ]

    rvec = @SVector [R00, R10, R01, R20, R11, R02]

    # The propagated covariance needs N⁻¹, so form it once and reuse it for
    # both the fitted coefficients and the output covariance.
    C = cholesky(Symmetric(Nmat))
    Ninv = SMatrix{6,6,T,36}(inv(C))

    X = Ninv * rvec
    a, b, c, d, e, f = X[1], X[2], X[3], X[4], X[5], X[6]

    # Centroid from the quadratic coefficients:
    #   D = [2d  e;  e  2f]
    #   x_c = (c e - 2 b f) / Δ,   y_c = (b e - 2 c d) / Δ,   Δ = 4 d f - e²
    two_d = 2 * d
    two_f = 2 * f
    Δ = two_d * two_f - e * e

    # Tikhonov-style regularisation for near-singular curvature matrix
    if abs(Δ) < T(1e-10)
        ε = T(1e-8)
        two_d += ε
        two_f += ε
        Δ = two_d * two_f - e * e
    end

    invΔ = inv(Δ)

    xc = (c * e - b * two_f) * invΔ
    yc = (b * e - c * two_d) * invΔ

    xc2 = xc * xc
    yc2 = yc * yc
    xcyc = xc * yc

    peak = a + b * xc + c * yc + d * xc2 + e * xcyc + f * yc2

    # Morphological diagnostics — roundness1_core (SROUND, bilateral vs.
    # fourfold symmetry) and roundness2_core (GROUND, marginal height ratio).
    # Both use the 8 neighbour pixels already in registers from the 3×3 patch.
    # DAOPHOT convention: 0 = symmetric/circular, nonzero = asymmetric/elongated.

    # SROUND on the 3×3 patch (DAOPHOT / photutils roundness1).
    # SUM2 = +45° axis sum minus -45° axis sum (bilateral asymmetry).
    # SUM4 = sum of absolute values over all 8 neighbours (fourfold normalization).
    sum2 = z12 + z32 - z21 - z23 + z11 + z33 - z31 - z13
    sum4 = abs(z12) + abs(z32) + abs(z21) + abs(z23) +
           abs(z11) + abs(z33) + abs(z31) + abs(z13)
    roundness1_core = if sum4 > eps(T)
        2 * sum2 / sum4
    else
        zero(T)
    end

    # Normalized curvature — negated Laplacian divided by peak value.
    # For a Gaussian, -(2d+2f)/peak ≈ 2/FWHM², independent of flux.
    normalized_curvature = -(two_d + two_f) / max(abs(peak), eps(T))

    # GROUND from the quadratic fit curvatures (DAOPHOT / photutils roundness2).
    # For a Gaussian, the marginal-fit height HX ∝ 1/σ_x ∝ √|d|, so
    # 2·(HX-HY)/(HX+HY) = 2·(√|d|-√|f|)/(√|d|+√|f|).
    dreg = two_d / 2
    freg = two_f / 2
    sqrt_ad = sqrt(abs(dreg))
    sqrt_af = sqrt(abs(freg))
    denom = sqrt_ad + sqrt_af
    roundness2_core = if denom > eps(T)
        2 * (sqrt_ad - sqrt_af) / denom
    else
        zero(T)
    end

    # Jacobian of (yc, xc, peak) w.r.t. (a, b, c, d, e, f).
    # At the stationary point ∂P/∂x = ∂P/∂y = 0, so d(peak)/dθ simplifies
    # to the basis vector evaluated at the centroid.
    J = @SMatrix [
        zero(T)   e * invΔ         -two_d * invΔ   -(2 * c + 4 * freg * yc) * invΔ   (b + 2 * e * yc) * invΔ   -4 * dreg * yc * invΔ
        zero(T)  -two_f * invΔ     e * invΔ        -4 * freg * xc * invΔ             (c + 2 * e * xc) * invΔ   -(2 * b + 4 * dreg * xc) * invΔ
        one(T)    xc               yc              xc2                               xcyc                      yc2
    ]

    cov = J * Ninv * transpose(J)

    y_err = sqrt(max(zero(T), cov[1,1]))
    x_err = sqrt(max(zero(T), cov[2,2]))
    peak_err = sqrt(max(zero(T), cov[3,3]))

    # Inverse-variance-weighted center-of-mass on the 3×3 patch.
    invR00 = inv(R00)
    com_x = R10 * invR00
    com_y = R01 * invR00

    # Delta-method covariance of the ratio estimator com = R / S00.
    # Var(R_pq) = S_{2p,2q}, Cov(R_pq, R_rs) = S_{p+r, q+s}.
    invR00_sq = invR00 * invR00
    var_com_x = (S20 - 2 * com_x * S10 + com_x * com_x * S00) * invR00_sq
    var_com_y = (S02 - 2 * com_y * S01 + com_y * com_y * S00) * invR00_sq
    cov_com_xy = (S11 - com_x * S01 - com_y * S10 + com_x * com_y * S00) * invR00_sq
    com_cov = @SMatrix [var_com_y cov_com_xy; cov_com_xy var_com_x]
    com_y_err = sqrt(max(zero(T), var_com_y))
    com_x_err = sqrt(max(zero(T), var_com_x))

    return (; poly = (; y = yc, x = xc, peak,
                      y_err, x_err, peak_err, cov),
             com = (; y = com_y, x = com_x,
                     cov = com_cov,
                     y_err = com_y_err, x_err = com_x_err),
             normalized_curvature, roundness1_core, roundness2_core)
end

"""
    centroid_poly(image::AbstractMatrix [, inv_var]) -> NamedTuple

Polynomial centroid of a point source in `image`. This version is
designed to work with image cutouts containing a single point source,
where the brightest pixel is expected to be near the source centroid.

Finds the brightest pixel via `findmax`, extracts the surrounding 3×3
patch, fits a 2nd-order 2-D polynomial via weighted least squares (see
[`_centroid_poly3`](@ref)), and returns both the polynomial centroid and
the inverse-variance-weighted center-of-mass (COM) centroid in *global*
pixel coordinates.

# Arguments
- `image::AbstractMatrix`: image cutout containing a point source.
- `inv_var::AbstractMatrix`: per-pixel inverse variance (same size as
  `image`).  If omitted, a uniform inverse variance `Fill(1, size(image))`
  is used (equivalent to ordinary least squares).  Set entries to zero to
  mask bad or saturated pixels.

# Returns
A `NamedTuple` with keys `(; poly, com, normalized_curvature, roundness1_core,
roundness2_core)` where

- `poly` — `NamedTuple` `(; y, x, peak, y_err, x_err, peak_err, cov)`
  with the polynomial centroid in global pixel coordinates (row, column),
  the fitted value at the centroid, 1-σ uncertainties, and 3×3 `SMatrix`
  covariance of `(y, x, peak)`.  Access as `result.poly.y`, `result.poly.x`, etc.
- `com` — `NamedTuple` `(; y, x, y_err, x_err, cov)` with the
  inverse-variance-weighted center-of-mass centroid, its 1-σ
  uncertainties, and its 2×2 `SMatrix` covariance.  Access as
  `result.com.y`, `result.com.x`, etc.
- `normalized_curvature` — negated Laplacian divided by the fitted peak
  value; ``\\approx 2/\\mathrm{FWHM}^2`` for a Gaussian.  This flux-independent
statistic is useful for distinguishing stars from
  cosmic rays and hot pixels.
- `roundness1_core` — DAOPHOT SROUND / photutils `roundness1`:
  ``2\\cdot\\Sigma_2/\\Sigma_4`` from the 8 neighbour pixels.
  0 = symmetric, nonzero = asymmetric.
- `roundness2_core` — DAOPHOT GROUND / photutils `roundness2`:
  ``2(\\sqrt{|d|} - \\sqrt{|f|})/(\\sqrt{|d|} + \\sqrt{|f|})``.
  0 = circular core, negative = extended in x (columns),
  positive = extended in y (rows).

If the brightest pixel lies on the image border (no full 3×3
neighbourhood), every field is `NaN`:

```julia
(; poly = (; y = NaN, x = NaN, peak = NaN,
            y_err = NaN, x_err = NaN, peak_err = NaN,
            cov = @SMatrix [NaN NaN NaN; NaN NaN NaN; NaN NaN NaN]),
   com = (; y = NaN, x = NaN, y_err = NaN, x_err = NaN,
          cov = @SMatrix [NaN NaN; NaN NaN]),
   normalized_curvature = NaN, roundness1_core = NaN, roundness2_core = NaN)
```

# Examples
```jldoctest
julia> using CrowdPhot: centroid_poly

julia> img = [0.1 0.3 0.1; 0.3 1.0 0.3; 0.1 0.3 0.1];

julia> result = centroid_poly(img);

julia> round(result.poly.x; digits=1), round(result.poly.y; digits=1)
(2.0, 2.0)

julia> round(result.com.x; digits=1), round(result.com.y; digits=1)
(2.0, 2.0)
```

# References
See [Vakili2016](@citet) for details.
"""
function centroid_poly(image::AbstractMatrix{T}, inv_var::AbstractMatrix = Fill(one(T), size(image))) where {T <: Real}
    _, maxidx = findmax(image)
    i0, j0 = Tuple(maxidx)  # row, column
    return centroid_poly(image, Int(i0), Int(j0), inv_var)
end
"""
    centroid_poly(image, i0::Int, j0::Int, inv_var) -> NamedTuple

Variant of [`centroid_poly`](@ref) that accepts pre-computed brightest-pixel
coordinates `i0, j0` (corresponding to pixel `image[i0, j0]`) instead of
calling `findmax` internally. Useful when the caller has already identified
the peak pixel (e.g. from a correlation map). `i0` is the row index
(y-coordinate) and `j0` is the column index (x-coordinate).

Returns the same `NamedTuple` as the two-argument form.
"""
function centroid_poly(image::AbstractMatrix{T}, i0::Int, j0::Int, inv_var::AbstractMatrix = Fill(one(T), size(image))) where {T <: Real}
    # check that a full 3×3 neighbourhood exists
    nan = T(NaN)
    nan3 = @SMatrix [nan nan nan; nan nan nan; nan nan nan]
    nan2 = @SMatrix [nan nan; nan nan]
    nancom = (; y = nan, x = nan, y_err = nan, x_err = nan, cov = nan2)
    if i0 < 2 || i0 > size(image, 1) - 1 || j0 < 2 || j0 > size(image, 2) - 1
        return (; poly = (; y = nan, x = nan, peak = nan,
                          y_err = nan, x_err = nan, peak_err = nan,
                          cov = nan3),
                 com = nancom,
                 normalized_curvature = nan, roundness1_core = nan, roundness2_core = nan)
    end

    # extract 3×3 views
    patch = view(image, i0-1:i0+1, j0-1:j0+1)
    wpatch = view(inv_var, i0-1:i0+1, j0-1:j0+1)

    # delegate to the 3×3 solver
    local_result = _centroid_poly3(patch, wpatch)

    # convert local → global coordinates
    # i0 is row (y), j0 is column (x)
    return (; poly = (; y = i0 + local_result.poly.y,
                       x = j0 + local_result.poly.x,
                       peak = local_result.poly.peak,
                       y_err = local_result.poly.y_err,
                       x_err = local_result.poly.x_err,
                       peak_err = local_result.poly.peak_err,
                       cov = local_result.poly.cov),
             com = (; y = i0 + local_result.com.y,
                     x = j0 + local_result.com.x,
                     y_err = local_result.com.y_err,
                     x_err = local_result.com.x_err,
                     cov = local_result.com.cov),
             normalized_curvature = local_result.normalized_curvature,
             roundness1_core = local_result.roundness1_core,
             roundness2_core = local_result.roundness2_core)
end

"""
    choose_centroid(result) -> NamedTuple

Given the `NamedTuple` returned by [`centroid_poly`](@ref) or
[`_centroid_poly3`](@ref), choose between the polynomial centroid
(`result.poly.y`, `result.poly.x`) and the center-of-mass centroid
(`result.com.y`, `result.com.x`).

The polynomial centroid is preferred for well-sampled data where the
3×3 patch has enough curvature for a reliable quadratic fit.  The COM
centroid is chosen when the polynomial's curvature matrix is nearly
singular (e.g. for very broad PSFs), indicated by a polynomial-vs-COM
variance ratio exceeding 10².

Returns `(; y, x, source)` where `source` is `:poly` or `:com`.

!!! note
    This heuristic detects curvature degeneracy but cannot detect
    quadratic model bias on undersampled data.  For undersampled
    images the matched-filter step in the detection pipeline broadens
    the PSF enough that the polynomial centroid is usually reliable.
    If you are centroiding raw (un-convolved) undersampled data,
    prefer the COM centroid directly.
"""
function choose_centroid(result)
    # If the polynomial covariance is more than 100× the COM covariance,
    # the curvature matrix is essentially degenerate → use COM.
    if result.poly.cov[2,2] > 100 * result.com.cov[2,2]
        return (; y = result.com.y, x = result.com.x, source = :com)
    else
        return (; y = result.poly.y, x = result.poly.x, source = :poly)
    end
end
