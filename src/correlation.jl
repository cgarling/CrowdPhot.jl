# correlation.jl — 2D correlation filtering for CrowdPhot.jl
#
# Extracted and adapted from ImageFiltering.jl (MIT-licensed, Tim Holy et al.).
# This file provides a self-contained FIR (finite impulse response) 2D correlation
# engine with automatic separable-kernel detection via SVD.  It depends only on
# LinearAlgebra (stdlib) — no OffsetArrays, no image ecosystem, no FFTW.

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    correlate(img::AbstractMatrix, kernel, [border=:replicate]) -> Matrix

Compute the 2D correlation of `img` with `kernel`, returning a matrix of the
same size as `img`.  The `kernel` is **not** flipped — this is correlation,
not convolution.

`kernel` may be:
- A 2D `AbstractMatrix` (e.g., a rendered PSF image, see [`render`](@ref)).
  Separability is automatically detected via SVD rank-1 checking;
  if separable, the computation is done in two 1D passes for better performance.
- A `Tuple` of two matrices representing pre-factored 1D kernels,
  e.g. `(col_vec, row_vec)` where `col_vec` is `kr×1` and `row_vec` is `1×kc`.

`border` controls how the image is extended at the edges:
- `:replicate` (default) — repeat the nearest border pixel.
- `:zero` — pad with zeros.

See also: [`correlate!`](@ref) for an in-place variant.
"""
function correlate(img::AbstractMatrix{T}, kernel, border::Symbol=:replicate) where {T}
    S = promote_type(T, _eltype_kernel(kernel))
    out = similar(img, S)
    correlate!(out, img, kernel, border)
end

"""
    correlate!(out::AbstractMatrix, img::AbstractMatrix, kernel, [border=:replicate]) -> out

In-place variant of [`correlate`](@ref).  Writes the correlation result into `out`,
which must have the same axes as `img`.
"""
function correlate!(out::AbstractMatrix, img::AbstractMatrix, kernel, border::Symbol=:replicate)
    axes(out) == axes(img) || throw(DimensionMismatch(
        "output axes $(axes(out)) must match image axes $(axes(img))"
    ))
    kern = _canonicalize(kernel)
    pt, pb, pl, pr = _total_padding(kern, border)
    padded = _padarray(img, border, pt, pb, pl, pr)
    _correlate!(out, padded, kern)
    out
end

# ---------------------------------------------------------------------------
# Kernel canonicalization — convert everything to Tuple form
# ---------------------------------------------------------------------------

_eltype_kernel(kernel::AbstractMatrix) = eltype(kernel)
_eltype_kernel(kernel::Tuple) = promote_type(map(_eltype_kernel, kernel)...)

"""
    _canonicalize(kernel) -> Tuple{Vararg{AbstractMatrix}}

Normalize a kernel into tuple-of-matrices form for the correlation scheduler.

- An already-factored `Tuple` is passed through unchanged.
- A 2D `AbstractMatrix` is tested for separability via SVD.  If the matrix has
  rank 1 (all singular values beyond the first are ≤ `sqrt(eps(T))`), it is
  split into `(col_factor, row_factor)`.  Otherwise it is wrapped as a
  pseudo-factored pair `(identity, kernel)` where the identity element is
  a 1×1 matrix `[1.0]` that the scheduler skips.

For an `m×n` kernel the convention for the correlation center is
`cr = (m+1)÷2, cc = (n+1)÷2`.  Callers must supply odd-sized kernels
so the center falls on an integer pixel.
"""
_canonicalize(kernel::Tuple) = kernel
_canonicalize(kernel::AbstractMatrix) = _tryfactor(kernel)

function _tryfactor(kernel::AbstractMatrix{T}) where {T}
    m, n = size(kernel)
    # 1×1 is trivially separable but SVD + scheduling overhead isn't worth it.
    m == n == 1 && return (_identity_kernel(T), kernel)

    # Use a dense Matrix to guarantee StridedMatrix for svd.
    F = svd(Matrix{T}(kernel))
    U, S, Vt = F.U, F.S, F.Vt

    separable = true
    EPS = sqrt(eps(real(eltype(S))))
    @inbounds for i in 2:length(S)
        separable &= (abs(S[i]) < EPS)
    end

    if !separable
        return (_identity_kernel(T), kernel)
    end

    s = S[1]
    ss = sqrt(s)
    # U[:, 1] is an m-vector → reshape to m×1 (filters rows / dim 1).
    # Vt[1, :] is an n-vector → reshape to 1×n (filters columns / dim 2).
    k1 = reshape(ss .* U[:, 1], m, 1)
    k2 = reshape(ss .* Vt[1, :], 1, n)
    return (k1, k2)
end

# Sentinel identity kernel.  A 1×1 matrix [1.0] — convolving with this is a no-op.
_identity_kernel(::Type{T}) where {T} = fill(one(float(T)), 1, 1)

_isidentity(k::AbstractMatrix) = size(k) == (1, 1) && k[1, 1] == 1

# ---------------------------------------------------------------------------
# Padding calculation
# ---------------------------------------------------------------------------

"""
    _padding_needed(kernel::AbstractMatrix) -> (top, bottom, left, right)

Return the number of pixels needed on each side to compute the full correlation
at every pixel of the image.  The kernel centre is assumed to be at
`((kr+1)÷2, (kc+1)÷2)` for a `kr×kc` kernel.
"""
function _padding_needed(kernel::AbstractMatrix)
    kr, kc = size(kernel)
    cr, cc = (kr + 1) ÷ 2, (kc + 1) ÷ 2
    return (cr - 1, kr - cr, cc - 1, kc - cc)
end

function _total_padding(kernels::Tuple, border::Symbol)
    pt, pb, pl, pr = 0, 0, 0, 0
    for k in kernels
        if !_isidentity(k)
            t, b, l, r = _padding_needed(k)
            pt += t; pb += b; pl += l; pr += r
        end
    end
    return (pt, pb, pl, pr)
end

function _total_padding(kernel::AbstractMatrix, border::Symbol)
    return _padding_needed(kernel)
end

# ---------------------------------------------------------------------------
# Border padding
# ---------------------------------------------------------------------------

"""
    _padarray(img, border, top, bottom, left, right) -> Matrix

Return a padded copy of `img` extended by the given number of pixels on each
side.  `border` must be `:replicate` or `:zero`.
"""
function _padarray(img::AbstractMatrix{T}, border::Symbol,
                   top::Int, bottom::Int, left::Int, right::Int) where {T}
    H, W = size(img)
    H2, W2 = H + top + bottom, W + left + right
    padded = similar(img, T, H2, W2)

    # Copy the interior
    padded[top+1:top+H, left+1:left+W] .= img

    if border == :zero
        # Corners are already zero from similar, just zero the strips.
        _zero_strip!(padded, 1:top,         1:W2)          # top
        _zero_strip!(padded, top+H+1:H2,    1:W2)          # bottom
        _zero_strip!(padded, top+1:top+H,   1:left)        # left
        _zero_strip!(padded, top+1:top+H,   left+W+1:W2)   # right
    elseif border == :replicate
        # Top and bottom strips
        for col in 1:W2
            # top
            for row in 1:top
                @inbounds padded[row, col] = padded[top+1, col]
            end
            # bottom
            for row in top+H+1:H2
                @inbounds padded[row, col] = padded[top+H, col]
            end
        end
        # Left and right strips (including the corner extension already done above)
        for row in 1:H2
            for col in 1:left
                @inbounds padded[row, col] = padded[row, left+1]
            end
            for col in left+W+1:W2
                @inbounds padded[row, col] = padded[row, left+W]
            end
        end
    else
        throw(ArgumentError("unknown border mode :$border; use :replicate or :zero"))
    end
    return padded
end

function _zero_strip!(A, rows, cols)
    @inbounds for c in cols, r in rows
        A[r, c] = zero(eltype(A))
    end
end

# ---------------------------------------------------------------------------
# Correlation scheduler — routes to 1D-separable or 2D-inseparable paths
# ---------------------------------------------------------------------------

# The padded image always carries enough extra rows and columns so that the
# origin (pixel (1,1) of the original image) sits at (pt+1, pl+1) in the
# padded array, where pt = sum of top paddings of all factors, etc.
# Because padding per-factor matches its kernel-radius (cr−1 top, cc−1 left),
# the general indexing formula
#
#     img[offset_row + row + j − cr,  offset_col + col + j − cc]
#
# simplifies to  img[row + j − 1,  col + j − 1]  when the image has been
# padded with exactly the right amount.  The cascade preserves this invariant
# by constructing intermediate buffers with the correct residual padding.

function _correlate!(out::AbstractMatrix, img::AbstractMatrix, kernel::Tuple{Any})
    _correlate_2d!(out, img, first(kernel))
end

function _correlate!(out::AbstractMatrix, img::AbstractMatrix, kernel::Tuple{Any,Any})
    k1, k2 = kernel
    if _isidentity(k1)
        _correlate_2d!(out, img, k2)
        return
    end
    # k1 is m×1 (filters rows), k2 is 1×n (filters columns).
    # Full padding: rows padded for k1, columns padded for k2.
    # After k1: output has H rows, but the full padded column width survives
    # because k2 still needs column padding.
    T = promote_type(eltype(out), eltype(k1), eltype(k2))
    H = size(out, 1)
    tmp = similar(img, T, H, size(img, 2))
    _correlate_1d!(tmp, img, k1)
    _correlate_1d!(out, tmp, k2)
end

# Three or more factors — shouldn't arise from _canonicalize but handle
# gracefully by folding left through sequential 1D passes.
function _correlate!(out::AbstractMatrix, img::AbstractMatrix, kernel::Tuple)
    T = promote_type(eltype(out), _eltype_kernel(kernel))
    nonid = filter(k -> !_isidentity(k), collect(kernel))
    isempty(nonid) && return copyto!(out, img)
    n = length(nonid)
    # Determine which factors pad rows vs columns, and size buffers accordingly.
    # After each row-filter pass the row count shrinks; after each col-filter
    # pass the column count shrinks.  Walk the list right-to-left to determine
    # the buffer sizes needed at each step.
    _, row_end, _, col_end = _total_padding(kernel, :replicate)  # border unused here
    H0, W0 = size(out)  # final output size
    # initial padded size
    src_rows = H0 + row_end + _bottom_padding(kernel)
    src_cols = W0 + col_end + _right_padding(kernel)
    buf_rows, buf_cols = src_rows, src_cols
    src = img
    # Build a scratch destination sized for the first intermediate result
    for (i, k) in enumerate(nonid)
        pt, pb, pl, pr = _padding_needed(k)
        if pt + pb > 0   # row filter
            buf_rows = buf_rows - pt - pb
        end
        if pl + pr > 0   # col filter
            buf_cols = buf_cols - pl - pr
        end
        dest = (i == n) ? out : similar(img, T, buf_rows, buf_cols)
        _correlate_1d!(dest, src, k)
        src = dest
    end
end

# Sum of bottom/right padding across a tuple (companion to _total_padding).
function _bottom_padding(kernel::Tuple)
    s = 0
    for k in kernel
        _isidentity(k) && continue
        s += _padding_needed(k)[2]
    end
    return s
end

function _right_padding(kernel::Tuple)
    s = 0
    for k in kernel
        _isidentity(k) && continue
        s += _padding_needed(k)[4]
    end
    return s
end

# ---------------------------------------------------------------------------
# Inner correlation loops
#
# Invariant: the input image `img` to each of these functions is already
# padded so that the "origin" (where the current kernel's center aligns with
# the first valid output pixel) is at img[1, 1].  Consequently the indexing
# uses `row + j - 1` rather than `row + j - center` — the padding absorbs the
# center offset.
# ---------------------------------------------------------------------------

"""
    _correlate_1d!(out, img, kernel)

Dispatch to a 1D correlation pass based on `kernel` shape.
- `m×1` → filters along rows (dimension 1).
- `1×n` → filters along columns (dimension 2).
- anything else → falls back to `_correlate_2d!`.
"""
function _correlate_1d!(out::AbstractMatrix, img::AbstractMatrix,
                        kernel::AbstractMatrix)
    kr, kc = size(kernel)
    if kr > 1 && kc == 1
        _correlate_rows!(out, img, kernel)
    elseif kr == 1 && kc > 1
        _correlate_cols!(out, img, kernel)
    else
        _correlate_2d!(out, img, kernel)
    end
end

function _correlate_rows!(out::AbstractMatrix{S}, img::AbstractMatrix,
                          kernel::AbstractMatrix) where {S}
    kr = size(kernel, 1)
    Ho, Wo = size(out)
    z = zero(S)
    Threads.@threads for col in 1:Wo
        @inbounds for row in 1:Ho
            acc = z
            for j in 1:kr
                acc += img[row + j - 1, col] * kernel[j, 1]
            end
            out[row, col] = acc
        end
    end
    out
end

function _correlate_cols!(out::AbstractMatrix{S}, img::AbstractMatrix,
                          kernel::AbstractMatrix) where {S}
    kc = size(kernel, 2)
    Ho, Wo = size(out)
    z = zero(S)
    Threads.@threads for col in 1:Wo
        @inbounds for row in 1:Ho
            acc = z
            for j in 1:kc
                acc += img[row, col + j - 1] * kernel[1, j]
            end
            out[row, col] = acc
        end
    end
    out
end

function _correlate_2d!(out::AbstractMatrix{S}, img::AbstractMatrix,
                        kernel::AbstractMatrix) where {S}
    kr, kc = size(kernel)
    Ho, Wo = size(out)
    z = zero(S)
    Threads.@threads for col in 1:Wo
        @inbounds for row in 1:Ho
            acc = z
            for kc_i in 1:kc, kr_i in 1:kr
                acc += img[row + kr_i - 1, col + kc_i - 1] * kernel[kr_i, kc_i]
            end
            out[row, col] = acc
        end
    end
    out
end
