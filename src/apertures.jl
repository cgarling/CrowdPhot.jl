# Circular aperture geometry for curves of growth, encircled energy, and
# fitting-region support masks.  All coordinates follow image convention: y
# (row) first, x (column) second.
#
# Exact pixel-circle overlap math is adapted from astropy/photutils under the
# BSD 3-clause license and Photometry.jl under the MIT license.

# ==============================================================================
# Overlap classification — cheap fast-path discriminant
# ==============================================================================

@enum OverlapFlag outside inside partial

# ==============================================================================
# CircularAperture
# ==============================================================================

"""
    CircularAperture(y, x, r)
    CircularAperture(; y, x, r)

A circular aperture centered at image position `(y, x)` with radius `r`.

All coordinates follow image-index convention: `y` is the row index, `x` is
the column index.  `r` is in detector-pixel units.

!!! note
    This type is internal and not yet part of the public API.
"""
Base.@kwdef struct CircularAperture{T}
    y::T
    x::T
    r::T
    function CircularAperture(y, x, r)
        T = promote_type(typeof(y), typeof(x), typeof(r))
        # T = float(T)
        return new{T}(T(y), T(x), T(r))
    end
end

# ==============================================================================
# Spatial support
# ==============================================================================

"""
    bounding_axes(ap::CircularAperture) -> (yrange, xrange)

Return the integer pixel-index ranges `(yrange, xrange)` that cover every pixel
with possibly nonzero overlap with `ap`.  The ranges are closed (inclusive) and
follow image-index convention (y-first).
"""
function bounding_axes(ap::CircularAperture{T}) where {T}
    FT = float(T)
    r = FT(ap.r)
    yc = FT(ap.y)
    xc = FT(ap.x)
    half = FT(0.5)

    _ymin = yc - r - half
    _xmin = xc - r - half
    ymin = isinteger(_ymin) ? ceil(Int, _ymin) + 1 : ceil(Int, _ymin)
    xmin = isinteger(_xmin) ? ceil(Int, _xmin) + 1 : ceil(Int, _xmin)
    ymax = ceil(Int, yc + r - half)
    xmax = ceil(Int, xc + r - half)
    return ymin:ymax, xmin:xmax
end

"""
    clipped_axes(ap::CircularAperture, image::AbstractMatrix) -> (yrange, xrange)

Return `bounding_axes(ap)` clipped to lie entirely within the axes of
`image`.  The return order is `(yrange, xrange)` (y-first).
"""
function clipped_axes(ap::CircularAperture, image::AbstractMatrix)
    yr, xr = bounding_axes(ap)
    ay, ax = axes(image)
    return max(first(yr), first(ay)):min(last(yr), last(ay)),
           max(first(xr), first(ax)):min(last(xr), last(ax))
end

# ==============================================================================
# Overlap classification
# ==============================================================================

"""
    _overlap_flag(ap::CircularAperture, i, j) -> OverlapFlag

Classify pixel `(i, j)` (row, col in image-index convention) relative to `ap`.

Returns `inside` when the pixel is wholly inside the circle, `outside` when it
is wholly outside, and `partial` when it straddles the boundary.
"""
@inline function _overlap_flag(ap::CircularAperture{T}, i::Integer, j::Integer) where {T}
    FT = float(T)
    dy = FT(i) - FT(ap.y)
    dx = FT(j) - FT(ap.x)
    d2 = dx^2 + dy^2
    r = FT(ap.r)
    dr = FT(sqrt(2) / 2) # corner-center distance of pixel
    d2 > (r + dr)^2 && return outside
    r > dr && d2 < (r - dr)^2 && return inside
    return partial
end

# ==============================================================================
# Weight-evaluation method types
# ==============================================================================

"""
    ExactOverlap()

Exact fractional pixel-circle overlap area via analytic integration.

Edge pixels receive a weight in ``[0, 1]`` equal to the fraction of the
pixel's area covered by the circle.
"""
struct ExactOverlap end

"""
    CenterOverlap()

Binary weight: 1.0 when the pixel *center* is inside the circle, 0.0 otherwise.

Fastest method; useful for hard support masks in fitting loops.
"""
struct CenterOverlap end

"""
    WholePixelOverlap()

Binary weight: 1.0 only when the *entire* pixel (all four corners) is inside
the circle, 0.0 otherwise.

More conservative than [`CenterOverlap`](@ref); matches the existing
`_pixel_wholly_inside` behavior in the empirical PSF builder.
"""
struct WholePixelOverlap end

"""
    SubpixelOverlap{N}()

Approximate fractional overlap using ``N \\times N`` subpixel quadrature.

Faster than [`ExactOverlap`](@ref) for large ``N`` on some geometries; less
accurate for a given pixel.
"""
struct SubpixelOverlap{N} end

# ==============================================================================
# Weight evaluation
# ==============================================================================

"""
    aperture_weight(ap::CircularAperture, i, j, method) -> float(T)

Return the overlap weight of pixel `(i, j)` with `ap` using `method`.

`i` is the row index (y), `j` is the column index (x), matching `image[i, j]`.
"""
function aperture_weight(ap::CircularAperture{T}, i::Integer, j::Integer, ::ExactOverlap) where {T}
    FT = float(T)
    flag = _overlap_flag(ap, i, j)
    flag == outside && return zero(FT)
    flag == inside  && return one(FT)
    return _circular_overlap_exact(FT(j) - FT(ap.x), FT(i) - FT(ap.y), FT(ap.r))
end

function aperture_weight(ap::CircularAperture{T}, i::Integer, j::Integer, ::CenterOverlap) where {T}
    FT = float(T)
    dx = FT(j) - FT(ap.x)
    dy = FT(i) - FT(ap.y)
    return (dx^2 + dy^2 < FT(ap.r)^2) ? one(FT) : zero(FT)
end

function aperture_weight(ap::CircularAperture{T}, i::Integer, j::Integer, ::WholePixelOverlap) where {T}
    FT = float(T)
    r2 = FT(ap.r)^2
    yc = FT(ap.y)
    xc = FT(ap.x)
    # Check all four pixel corners.
    @inbounds for di in (-FT(0.5), FT(0.5)), dj in (-FT(0.5), FT(0.5))
        (FT(i) + di - yc)^2 + (FT(j) + dj - xc)^2 > r2 && return zero(FT)
    end
    return one(FT)
end

function aperture_weight(ap::CircularAperture{T}, i::Integer, j::Integer, ::SubpixelOverlap{N}) where {T, N}
    FT = float(T)
    return _circular_overlap_subpixel(
        FT(j) - FT(ap.x), FT(i) - FT(ap.y), FT(ap.r), N)
end

# ==============================================================================
# Exact pixel-circle overlap (ported from astropy/photutils, BSD 3-clause)
# ==============================================================================

"""
    _circular_overlap_exact(dx, dy, r) -> float(typeof(dx))

Exact fractional area of overlap between a unit pixel centered at offset
`(dx, dy)` from the circle center and a circle of radius `r`.

`dx` is the x-offset (column), `dy` is the y-offset (row).
"""
function _circular_overlap_exact(dx::Real, dy::Real, r::Real)
    R = float(promote_type(typeof(dx), typeof(dy), typeof(r)))
    r <= 0 && return zero(R)
    xmin = dx - R(0.5)
    xmax = dx + R(0.5)
    ymin = dy - R(0.5)
    ymax = dy + R(0.5)
    return _circular_overlap_single_exact(xmin, ymin, xmax, ymax, r)
end

"""
    _circular_overlap_single_exact(xmin, ymin, xmax, ymax, r)

Area of overlap between rectangle ``[xmin, xmax] × [ymin, ymax]`` and a
circle of radius `r` centered at the origin.
"""
function _circular_overlap_single_exact(xmin, ymin, xmax, ymax, r)
    R = float(promote_type(typeof(xmin), typeof(ymin), typeof(xmax), typeof(ymax), typeof(r)))
    r <= 0 && return zero(R)
    if 0 <= xmin
        0 <= ymin && return _circular_overlap_core(xmin, ymin, xmax, ymax, r)
        0 >= ymax && return _circular_overlap_core(-ymax, xmin, -ymin, xmax, r)
        return (_circular_overlap_single_exact(xmin, ymin, xmax, 0, r) +
                _circular_overlap_single_exact(xmin, 0, xmax, ymax, r))
    elseif 0 >= xmax
        0 <= ymin && return _circular_overlap_core(-xmax, ymin, -xmin, ymax, r)
        0 >= ymax && return _circular_overlap_core(-xmax, -ymax, -xmin, -ymin, r)
        return (_circular_overlap_single_exact(xmin, ymin, xmax, 0, r) +
                _circular_overlap_single_exact(xmin, 0, xmax, ymax, r))
    else
        0 <= ymin && return (_circular_overlap_single_exact(xmin, ymin, 0, ymax, r) +
                            _circular_overlap_single_exact(0, ymin, xmax, ymax, r))
        0 >= ymax && return (_circular_overlap_single_exact(xmin, ymin, 0, ymax, r) +
                            _circular_overlap_single_exact(0, ymin, xmax, ymax, r))
        return (_circular_overlap_single_exact(xmin, ymin, 0, 0, r) +
                _circular_overlap_single_exact(0, ymin, xmax, 0, r) +
                _circular_overlap_single_exact(xmin, 0, 0, ymax, r) +
                _circular_overlap_single_exact(0, 0, xmax, ymax, r))
    end
end

"""
    _circular_overlap_core(xmin, ymin, xmax, ymax, r)

Core circular-overlap routine.  Assumes ``0 ≤ xmin`` and ``0 ≤ ymin``
(first-quadrant symmetry reduction).
"""
function _circular_overlap_core(xmin, ymin, xmax, ymax, r)
    R = float(promote_type(typeof(xmin), typeof(ymin), typeof(xmax), typeof(ymax), typeof(r)))
    xmin^2 + ymin^2 > r^2 && return zero(R)
    xmax^2 + ymax^2 < r^2 && return R((xmax - xmin) * (ymax - ymin))

    d1 = sqrt(xmax^2 + ymin^2)
    d2 = sqrt(xmin^2 + ymax^2)
    if d1 < r && d2 < r
        x1, y1 = sqrt(r^2 - ymax^2), ymax
        x2, y2 = xmax, sqrt(r^2 - xmax^2)
        area = ((xmax - xmin) * (ymax - ymin) -
                _area_triangle(x1, y1, x2, y2, xmax, ymax) +
                _area_arc(x1, y1, x2, y2, r))
    elseif d1 < r
        x1, y1 = xmin, sqrt(r^2 - xmin^2)
        x2, y2 = xmax, sqrt(r^2 - xmax^2)
        area = (_area_arc(x1, y1, x2, y2, r) +
                _area_triangle(x1, y1, x1, ymin, xmax, ymin) +
                _area_triangle(x1, y1, x2, ymin, x2, y2))
    elseif d2 < r
        x1, y1 = sqrt(r^2 - ymin^2), ymin
        x2, y2 = sqrt(r^2 - ymax^2), ymax
        area = (_area_arc(x1, y1, x2, y2, r) +
                _area_triangle(x1, y1, xmin, y1, xmin, ymax) +
                _area_triangle(x1, y1, xmin, y2, x2, y2))
    else
        x1, y1 = sqrt(r^2 - ymin^2), ymin
        x2, y2 = xmin, sqrt(r^2 - xmin^2)
        area = (_area_arc(x1, y1, x2, y2, r) +
                _area_triangle(x1, y1, x2, y2, xmin, ymin))
    end

    return R(area)
end

# ==============================================================================
# Geometric helpers
# ==============================================================================

"""Area of a triangle defined by three vertices."""
@inline _area_triangle(x0, y0, x1, y1, x2, y2) =
    abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)) / 2

"""
Area of a circular segment above a chord between two points with circle
radius `r`.

[Reference](http://mathworld.wolfram.com/CircularSegment.html)
"""
@inline function _area_arc(x0, y0, x1, y1, r)
    a = sqrt((x1 - x0)^2 + (y1 - y0)^2)
    θ = 2 * asin(a / (2 * r))
    return r^2 * (θ - sin(θ)) / 2
end

# ==============================================================================
# Subpixel quadrature
# ==============================================================================

"""
    _circular_overlap_subpixel(dx, dy, r, subpixels)

Approximate fractional overlap using ``subpixels × subpixels`` quadrature.

`dx`, `dy` are the pixel-center offsets from the circle center.
"""
function _circular_overlap_subpixel(dx::Real, dy::Real, r::Real, subpixels::Integer)
    FT = float(promote_type(typeof(dx), typeof(dy), typeof(r)))
    r <= 0 && return zero(FT)
    xmin = FT(dx) - FT(0.5)
    xmax = FT(dx) + FT(0.5)
    ymin = FT(dy) - FT(0.5)
    ymax = FT(dy) + FT(0.5)
    r2 = FT(r)^2

    frac = 0
    ddx = (xmax - xmin) / subpixels
    ddy = (ymax - ymin) / subpixels

    x = xmin + ddx / 2
    for _ in 1:subpixels
        y = ymin + ddy / 2
        for _ in 1:subpixels
            if x^2 + y^2 < r2
                frac += 1
            end
            y += ddy
        end
        x += ddx
    end
    return FT(frac) / FT(subpixels^2)
end
