using StaticArrays: SMatrix, @SMatrix, @SVector
using FillArrays: Fill
using LinearAlgebra: Symmetric, cholesky

# ---------------------------------------------------------------------------
# 3x3 polynomial centroiding  —  Vakili & Hogg (2016), arXiv:1610.05873
#
# The 3x3 patch is indexed in row-major order with local coordinates
# (x, y) where x is the column offset and y is the row offset, each
# in {-1, 0, 1} relative to the centre pixel:
#
#   (-1,-1)  (0,-1)  (1,-1)
#   (-1, 0)  (0, 0)  (1, 0)
#   (-1, 1)  (0, 1)  (1, 1)
#
# Design-matrix row for pixel i: [1, x_i, y_i, x_i², x_i*y_i, y_i²]
#
# The normal-equation matrix N = Aᵀ W A and right-hand side Aᵀ W z are built
# in closed form from weighted geometric moments S_pq = Σ w_i x_i^p y_i^q,
# exploiting that x,y ∈ {-1,0,1} collapses higher powers (x³=x, x⁴=x², …).
# ---------------------------------------------------------------------------

"""
    _centroid_poly3(image, inv_var) -> NamedTuple

Fit a 2nd-order 2-D polynomial ``P(x,y) = a + bx + cy + dx² + exy + fy²``
to a 3×3 patch using weighted least squares with inverse-variance weights
`inv_var`.  Returns `(; x, y, peak, x_err, y_err, peak_err, cov)` where
`x, y` are the sub-pixel centroid coordinates relative to the patch centre,
`peak` is the polynomial value at the centroid, the `_err` fields are 1-σ
uncertainties, and `cov` is the full 3×3 ``SMatrix`` covariance of
``(x, y, peak)``.

The design matrix is fixed (local coordinates `{-1,0,1}²`), so the
only free inputs are the 9 pixel values and 9 inverse-variance weights.

If the curvature matrix ``D = [2d  e;  e  2f]`` is near-singular
(``|4df - e²| < 10^{-10}``), a small Tikhonov-style regularisation is
added to its diagonal before computing the centroid.  If the data are
so noisy that the regularised determinant is still effectively zero,
`cov` will be large but the centroid estimates remain finite.
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

    # Use the regularised curvature values consistently in the propagated
    # derivatives (only matters when the near-singular branch was taken).
    dreg = two_d / 2
    freg = two_f / 2

    # Jacobian of (xc, yc, peak) w.r.t. (a, b, c, d, e, f).
    # At the stationary point ∂P/∂x = ∂P/∂y = 0, so d(peak)/dθ simplifies
    # to the basis vector evaluated at the centroid.
    J = @SMatrix [
        zero(T)  -two_f * invΔ                     e * invΔ        -4 * freg * xc * invΔ              (c + 2 * e * xc) * invΔ   -(2 * b + 4 * dreg * xc) * invΔ
        zero(T)   e * invΔ                        -two_d * invΔ    -(2 * c + 4 * freg * yc) * invΔ     (b + 2 * e * yc) * invΔ   -4 * dreg * yc * invΔ
        one(T)    xc                               yc              xc2                                xcyc                        yc2
    ]

    cov = J * Ninv * transpose(J)

    x_err = sqrt(max(zero(T), cov[1,1]))
    y_err = sqrt(max(zero(T), cov[2,2]))
    peak_err = sqrt(max(zero(T), cov[3,3]))

    return (; x = xc, y = yc, peak, x_err, y_err, peak_err, cov)
end

"""
    centroid_poly(image, inv_var = nothing) -> NamedTuple

Polynomial centroid of a point source in `image`.

Finds the brightest pixel, extracts the surrounding 3×3 patch, fits a
2nd-order 2-D polynomial via weighted least squares (see
[`_centroid_poly3`](@ref)), and returns the centroid in *global* pixel
coordinates together with the fitted peak and covariance.

If the brightest pixel lies on the image border (no full 3×3
neighbourhood), all fields are ``NaN``.

# Arguments
- `image::AbstractMatrix`: image cutout containing a point source.
- `inv_var::AbstractMatrix`: per-pixel inverse variance (same size as
  `image`).  If omitted, a uniform inverse variance `Fill(1, size(image))`
  is used (equivalent to ordinary least squares).

# Returns
`(; x, y, peak, x_err, y_err, peak_err, cov)` — centroid in global pixel
coordinates, fitted peak, 1-σ uncertainties, and the full 3×3 ``SMatrix``
covariance of ``(x, y, peak)``.

# Examples
```jldoctest
julia> using CrowdPhot

julia> img = [0.1 0.3 0.1; 0.3 1.0 0.3; 0.1 0.3 0.1];

julia> result = centroid_poly(img);

julia> round(result.x; digits=1), round(result.y; digits=1)
(2.0, 2.0)
```
"""
function centroid_poly(image::AbstractMatrix{T}, inv_var::AbstractMatrix = Fill(one(T), size(image))) where {T <: Real}
    _, maxidx = findmax(image)
    i0, j0 = Tuple(maxidx)  # row, column
    return centroid_poly(image, Int(i0), Int(j0), inv_var)
end
"""
    centroid_poly(image::AbstractMatrix{T}, i0::Int, j0::Int, inv_var::AbstractMatrix = Fill(one(T), size(image)))
This version of `centroid_poly` allows the caller to specify the brightest pixel coordinates `i0, j0` (row, column)
instead of computing them internally. This can be useful if the caller has already identified the brightest pixel
(e.g., by running a maximum filter over a detection image) and wants to avoid redundant computations.
"""
function centroid_poly(image::AbstractMatrix{T}, i0::Int, j0::Int, inv_var::AbstractMatrix = Fill(one(T), size(image))) where {T <: Real}
    # check that a full 3×3 neighbourhood exists
    nan = T(NaN)
    nan3 = @SMatrix [nan nan nan; nan nan nan; nan nan nan]
    if i0 < 2 || i0 > size(image, 1) - 1 || j0 < 2 || j0 > size(image, 2) - 1
        return (; x = nan, y = nan, peak = nan,
                 x_err = nan, y_err = nan, peak_err = nan,
                 cov = nan3)
    end

    # extract 3×3 views
    patch = view(image, i0-1:i0+1, j0-1:j0+1)
    wpatch = view(inv_var, i0-1:i0+1, j0-1:j0+1)

    # delegate to the 3×3 solver
    local_result = _centroid_poly3(patch, wpatch)

    # convert local → global coordinates
    # local x is column offset, local y is row offset
    return (; x = j0 + local_result.x,
             y = i0 + local_result.y,
             peak = local_result.peak,
             x_err = local_result.x_err,
             y_err = local_result.y_err,
             peak_err = local_result.peak_err,
             cov = local_result.cov)
end
