using StaticArrays: SMatrix, SVector, MMatrix, MVector, @SMatrix
using FillArrays: Fill
using LinearAlgebra: Symmetric, cholesky, Cholesky

# ---------------------------------------------------------------------------
# Precomputed design matrix for 3x3 polynomial fit.
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
# ---------------------------------------------------------------------------
const _DESIGN_3X3 = @SMatrix [
    1  -1  -1   1   1   1    # (-1,-1)
    1   0  -1   0   0   1    # ( 0,-1)
    1   1  -1   1  -1   1    # ( 1,-1)
    1  -1   0   1   0   0    # (-1, 0)
    1   0   0   0   0   0    # ( 0, 0)
    1   1   0   1   0   0    # ( 1, 0)
    1  -1   1   1  -1   1    # (-1, 1)
    1   0   1   0   0   1    # ( 0, 1)
    1   1   1   1   1   1    # ( 1, 1)
]

"""
    _centroid_poly3(image, inv_var) -> NamedTuple

Fit a 2nd-order 2-D polynomial ``P(x,y) = a + bx + cy + dx² + exy + fy²``
to a 3×3 patch using weighted least squares with inverse-variance weights
`inv_var`.  Returns `(; x, y, peak, x_err, y_err, peak_err)` where `x, y`
are the sub-pixel centroid coordinates relative to the patch centre,
`peak` is the polynomial value at the centroid, and the `_err` fields are
1-σ uncertainties propagated from the parameter covariance matrix.

The design matrix is fixed (local coordinates `{-1,0,1}²`), so the
only free inputs are the 9 pixel values and 9 inverse-variance weights.

If the curvature matrix ``D = [2d  e;  e  2f]`` is near-singular
(``|4df - e²| < 10^{-10}``), a small Tikhonov-style regularisation is
added to its diagonal before computing the centroid.  If the data are
so noisy that the regularised determinant is still effectively zero,
`(x_err, y_err, peak_err)` will be large but the centroid estimates
remain finite.
"""
function _centroid_poly3(image::AbstractMatrix{T}, inv_var::AbstractMatrix{T}) where {T <: Real}
    # --- flatten the 3×3 patch in row-major order --------------------------------
    z = SVector{9,T}(
        image[1,1], image[1,2], image[1,3],
        image[2,1], image[2,2], image[2,3],
        image[3,1], image[3,2], image[3,3],
    )
    w = SVector{9,T}(
        inv_var[1,1], inv_var[1,2], inv_var[1,3],
        inv_var[2,1], inv_var[2,2], inv_var[2,3],
        inv_var[3,1], inv_var[3,2], inv_var[3,3],
    )

    # --- build weighted normal equations  N = A' W A,  r = A' W z -----------------
    # Use mutable buffers and a single explicit loop so the compiler can
    # unroll everything for the statically-known 9×6 design.
    N = MMatrix{6,6,T,36}(undef)
    r = MVector{6,T}(undef)
    fill!(N, zero(T))
    fill!(r, zero(T))

    @inbounds for i in 1:9
        wi = w[i]
        iszero(wi) && continue   # zero-weight pixels contribute nothing
        zi = z[i]
        row_i = _DESIGN_3X3[i, :]  # SVector{6,Float64}
        # accumulate r = Σ w_i z_i a_i
        r1 = muladd(wi * zi, row_i[1], r[1])
        r2 = muladd(wi * zi, row_i[2], r[2])
        r3 = muladd(wi * zi, row_i[3], r[3])
        r4 = muladd(wi * zi, row_i[4], r[4])
        r5 = muladd(wi * zi, row_i[5], r[5])
        r6 = muladd(wi * zi, row_i[6], r[6])
        r[1] = r1; r[2] = r2; r[3] = r3
        r[4] = r4; r[5] = r5; r[6] = r6
        # accumulate N = Σ w_i a_i a_i'  (symmetric, fill upper triangle)
        wr1 = wi * row_i[1]
        wr2 = wi * row_i[2]
        wr3 = wi * row_i[3]
        wr4 = wi * row_i[4]
        wr5 = wi * row_i[5]
        wr6 = wi * row_i[6]
        N[1,1] += wr1 * row_i[1]
        N[1,2] += wr1 * row_i[2];  N[2,2] += wr2 * row_i[2]
        N[1,3] += wr1 * row_i[3];  N[2,3] += wr2 * row_i[3];  N[3,3] += wr3 * row_i[3]
        N[1,4] += wr1 * row_i[4];  N[2,4] += wr2 * row_i[4];  N[3,4] += wr3 * row_i[4]
        N[4,4] += wr4 * row_i[4]
        N[1,5] += wr1 * row_i[5];  N[2,5] += wr2 * row_i[5];  N[3,5] += wr3 * row_i[5]
        N[4,5] += wr4 * row_i[5];  N[5,5] += wr5 * row_i[5]
        N[1,6] += wr1 * row_i[6];  N[2,6] += wr2 * row_i[6];  N[3,6] += wr3 * row_i[6]
        N[4,6] += wr4 * row_i[6];  N[5,6] += wr5 * row_i[6];  N[6,6] += wr6 * row_i[6]
    end
    # fill lower triangle
    N[2,1] = N[1,2]
    N[3,1] = N[1,3];  N[3,2] = N[2,3]
    N[4,1] = N[1,4];  N[4,2] = N[2,4];  N[4,3] = N[3,4]
    N[5,1] = N[1,5];  N[5,2] = N[2,5];  N[5,3] = N[3,5];  N[5,4] = N[4,5]
    N[6,1] = N[1,6];  N[6,2] = N[2,6];  N[6,3] = N[3,6];  N[6,4] = N[4,6];  N[6,5] = N[5,6]

    # --- solve the 6×6 system ----------------------------------------------------
    Nmat = SMatrix{6,6,T,36}(N)
    rvec = SVector{6,T}(r)

    # Cholesky factorisation for symmetric-positive-definite Nmat
    C = cholesky(Symmetric(Nmat))
    X = C \ rvec
    a, b, c, d, e, f = X[1], X[2], X[3], X[4], X[5], X[6]

    # --- centroid from the quadratic coefficients --------------------------------
    # D = [2d  e;  e  2f]    →    x_c = (c e - 2 b f) / Δ,   y_c = (b e - 2 c d) / Δ
    two_d = 2d
    two_f = 2f
    Δ = two_d * two_f - e * e   # = 4df - e²

    # soft regularisation when curvature is nearly flat
    if abs(Δ) < 1e-10
        ε = T(1e-8)
        two_d += ε
        two_f += ε
        Δ = two_d * two_f - e * e
    end

    invΔ = inv(Δ)
    xc = (c * e - b * two_f) * invΔ
    yc = (b * e - c * two_d) * invΔ

    # peak value of the polynomial at the centroid
    peak = a + b * xc + c * yc + d * xc^2 + e * xc * yc + f * yc^2

    # --- error propagation -------------------------------------------------------
    # Covariance of parameters:  Cov(X) = N⁻¹
    Cinv = SMatrix{6,6,T,36}(inv(C))  # C is the Cholesky; inv(C) gives N⁻¹

    # Jacobian of (xc, yc, peak) w.r.t. (a, b, c, d, e, f)
    # ∂xc/∂a = 0,  ∂yc/∂a = 0,  ∂peak/∂a = 1
    J11 = zero(T)                  # ∂xc/∂a
    J12 = -two_f * invΔ            # ∂xc/∂b
    J13 =  e     * invΔ            # ∂xc/∂c
    J14 = -4f   * xc * invΔ        # ∂xc/∂d
    J15 = (c + 2e * xc) * invΔ     # ∂xc/∂e
    J16 = -(2b + 4d * xc) * invΔ   # ∂xc/∂f

    J21 = zero(T)                  # ∂yc/∂a
    J22 =  e     * invΔ            # ∂yc/∂b
    J23 = -two_d * invΔ            # ∂yc/∂c
    J24 = -(2c + 4f * yc) * invΔ   # ∂yc/∂d
    J25 = (b + 2e * yc) * invΔ     # ∂yc/∂e
    J26 = -4d * yc * invΔ          # ∂yc/∂f

    J31 = one(T)   # ∂peak/∂a
    J32 = xc       # ∂peak/∂b
    J33 = yc       # ∂peak/∂c
    J34 = xc^2     # ∂peak/∂d
    J35 = xc * yc  # ∂peak/∂e
    J36 = yc^2     # ∂peak/∂f

    J = @SMatrix [
        J11 J12 J13 J14 J15 J16
        J21 J22 J23 J24 J25 J26
        J31 J32 J33 J34 J35 J36
    ]

    # Cov(xc, yc, peak) = J * Cov(X) * J'
    Cov = J * Cinv * transpose(J)

    x_err = sqrt(max(zero(T), Cov[1,1]))
    y_err = sqrt(max(zero(T), Cov[2,2]))
    peak_err = sqrt(max(zero(T), Cov[3,3]))

    return (; x = xc, y = yc, peak,
             x_err, y_err, peak_err)
end

"""
    centroid_poly(image, inv_var = nothing) -> NamedTuple

Polynomial centroid of a point source in `image`.

Finds the brightest pixel, extracts the surrounding 3×3 patch, fits a
2nd-order 2-D polynomial via weighted least squares (see
[`_centroid_poly3`](@ref)), and returns the centroid in *global* pixel
coordinates together with the fitted peak and 1-σ uncertainties.

If the brightest pixel lies on the image border (no full 3×3
neighbourhood), `(; x=NaN, y=NaN, peak=NaN, x_err=NaN, y_err=NaN, peak_err=NaN)`
is returned.

# Arguments
- `image::AbstractMatrix`: image cutout containing a point source.
- `inv_var::AbstractMatrix`: per-pixel inverse variance (same size as
  `image`).  If omitted, a uniform inverse variance `Fill(1, size(image))`
  is used (equivalent to ordinary least squares).

# Returns
`(; x, y, peak, x_err, y_err, peak_err)` — centroid in global pixel
coordinates, fitted peak, and 1-σ uncertainties propagated from the
weighted least-squares parameter covariance.

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
    # locate brightest pixel (first occurrence if ties)
    maxval, maxidx = findmax(image)
    i0, j0 = Tuple(maxidx)  # row, column

    # check that a full 3×3 neighbourhood exists
    if i0 < 2 || i0 > size(image, 1) - 1 || j0 < 2 || j0 > size(image, 2) - 1
        nan = T(NaN)
        return (; x = nan, y = nan, peak = nan,
                 x_err = nan, y_err = nan, peak_err = nan)
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
             peak_err = local_result.peak_err)
end
