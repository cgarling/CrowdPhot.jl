"""
    GriddedPSFModel(psfs, grid_y, grid_x; y=0, x=0, flux=1, bkg=0)

Spatially-varying PSF model defined by a rectangular grid of "node" PSF
models tabulated at fiducial detector positions `(grid_y[i], grid_x[i])`.
`psfs` may be composed of any concrete `AbstractPSFModel` subtype (e.g. [`ImagePSF`](@ref),
[`GaussianPRF`](@ref)); the PSF shape at an arbitrary `(y, x)` detector
position is obtained via bilinear interpolation between the (up to four)
nearest node models, following the algorithm used by
`photutils.psf.GriddedPSFModel`.

`grid_y` and `grid_x` must form a complete rectangular grid, i.e. every
combination of the unique `grid_y` and unique `grid_x` values must be present
exactly once, unless only a single node PSF is supplied (`length(psfs) ==
1`), in which case that PSF is used everywhere (no interpolation). Node
positions do not need to be uniformly spaced.

`y` and `x` give the detector position at which the model is evaluated,
`flux` is the overall flux scaling factor, and `bkg` is a scalar background
level; these four fields are the only fittable parameters (analogous to
[`ImagePSF`](@ref)'s `y`, `x`, `flux`, `bkg`) -- the node PSFs and grid
positions are fixed.

# Node model requirements
Every node PSF must support `ConstructionBase.setproperties` with `y`, `x`,
`flux`, and `bkg` keys, and (if used with [`fit_star`](@ref)) `evaluate_fg`
whose gradient orders `y` and `x` first and `flux` and `bkg` last. This is the
convention followed by every `AbstractPSFModel` in this package. All node
PSFs must share the same element type `T`.

!!! note
    Unlike `photutils.psf.GriddedPSFModel`, this type has no `oversampling`,
    `origin`, or `fill_value` fields of its own: those are properties of how
    an individual node PSF is tabulated and evaluated (e.g. [`ImagePSF`](@ref)
    already carries its own `oversampling`/`origin`/`fill_value`), so they
    are naturally handled by each node rather than duplicated at the grid
    level.

```jldoctest
julia> using CrowdPhot.PSF: GriddedPSFModel, CircularGaussianPRF

julia> nodes = [CircularGaussianPRF(y=y0, x=x0, fwhm=3.0, flux=1.0, bkg=0.0)
                for (y0, x0) in ((0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0))];

julia> model = GriddedPSFModel(nodes, [0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 0.0, 10.0]; y=5.0, x=5.0, flux=100.0, bkg=1.0);

julia> model isa GriddedPSFModel{Float64, CircularGaussianPRF{Float64}}
true
```
"""
struct GriddedPSFModel{T, M <: AbstractPSFModel{T}} <: AbstractPSFModel{T}
    psfs::Vector{M}
    grid_y::Vector{T}
    grid_x::Vector{T}
    ygrid::Vector{T}
    xgrid::Vector{T}
    index_grid::Matrix{Int}
    y::T
    x::T
    flux::T
    bkg::T
end

function GriddedPSFModel(
        psfs::AbstractVector{<:AbstractPSFModel},
        grid_y::AbstractVector,
        grid_x::AbstractVector;
        y = 0,
        x = 0,
        flux = 1,
        bkg = 0
    )
    N = length(psfs)
    N ≥ 1 || throw(ArgumentError("`psfs` must be non-empty"))
    length(grid_y) == N && length(grid_x) == N ||
        throw(ArgumentError("`psfs`, `grid_y`, and `grid_x` must all have the same length"))

    # All node PSFs must share one element type
    # TODO: Node PSFs with heterogeneous element types are currently rejected; a future
    # version could support promoting/converting node PSFs to a common wider type
    # at construction time.
    T = eltype(psfs[1])
    all(m -> eltype(m) === T, psfs) ||
        throw(ArgumentError("all `psfs` must share the same element type `T`; got element types $(unique(eltype.(psfs)))"))

    M = eltype(psfs)
    M <: AbstractPSFModel{T} || throw(
        ArgumentError(
            "element type of `psfs` ($M) must be a subtype of `AbstractPSFModel{$T}`; " *
                "if mixing node model types, construct `psfs` as `AbstractPSFModel{$T}[...]`"
        )
    )

    all(isfinite, grid_y) && all(isfinite, grid_x) ||
        throw(ArgumentError("`grid_y` and `grid_x` must be finite"))

    # Validate that node models follow the package-wide (y, x, ..., flux, bkg)
    # property ordering; this is relied upon generically by `evaluate_fg`.
    keys1 = keys(ConstructionBase.getproperties(psfs[1]))
    length(keys1) ≥ 4 && keys1[1] === :y && keys1[2] === :x && keys1[end - 1] === :flux && keys1[end] === :bkg ||
        throw(
        ArgumentError(
            "node PSF models must expose properties ordered (y, x, ..., flux, bkg) " *
                "via `ConstructionBase.getproperties`; got $keys1"
        )
    )

    psfs_owned = Vector{M}(psfs)
    gy = Vector{T}(grid_y)
    gx = Vector{T}(grid_x)

    ygrid = sort(unique(gy))
    xgrid = sort(unique(gx))
    ny, nx = length(ygrid), length(xgrid)

    if N == 1
        index_grid = reshape([1], 1, 1)
    else
        (nx ≥ 2 && ny ≥ 2) ||
            throw(ArgumentError("`grid_y` and `grid_x` must each contain at least 2 unique values unless a single PSF is supplied"))
        nx * ny == N || throw(
            ArgumentError(
                "`grid_y`/`grid_x` positions do not form a complete rectangular grid " *
                    "($N points supplied; a $ny x $nx grid requires $(ny * nx))"
            )
        )
        index_grid = zeros(Int, ny, nx)
        for i in 1:N
            iy = searchsortedfirst(ygrid, gy[i])
            ix = searchsortedfirst(xgrid, gx[i])
            index_grid[iy, ix] == 0 ||
                throw(ArgumentError("duplicate node position at (y=$(gy[i]), x=$(gx[i]))"))
            index_grid[iy, ix] = i
        end
        all(!iszero, index_grid) ||
            throw(ArgumentError("`grid_y`/`grid_x` positions do not form a complete rectangular grid"))
    end

    return GriddedPSFModel{T, M}(psfs_owned, gy, gx, ygrid, xgrid, index_grid, T(y), T(x), T(flux), T(bkg))
end

ConstructionBase.getproperties(model::GriddedPSFModel) = (y = model.y, x = model.x, flux = model.flux, bkg = model.bkg)

function ConstructionBase.setproperties(model::GriddedPSFModel{T, M}, patch::NamedTuple) where {T, M}
    # Only the fit parameters can change; the node PSFs and grid stay fixed.
    y = haskey(patch, :y) ? T(patch.y) : model.y
    x = haskey(patch, :x) ? T(patch.x) : model.x
    flux = haskey(patch, :flux) ? T(patch.flux) : model.flux
    bkg = haskey(patch, :bkg) ? T(patch.bkg) : model.bkg
    return GriddedPSFModel{T, M}(
        model.psfs, model.grid_y, model.grid_x, model.ygrid, model.xgrid,
        model.index_grid, y, x, flux, bkg
    )
end

"""
    _grid_corners_dw(model::GriddedPSFModel{T}, Y, X) -> NTuple{4, Tuple{Int, T, T, T}}

Like [`_grid_corners`](@ref) but each element also carries `(dw/dY, dw/dX)`,
the partial derivatives of the bilinear weight with respect to the query
position. Used by `evaluate_fg`. Derivatives are zero for a position that is
clamped (i.e. outside the grid, or exactly on its boundary) in that
dimension, consistent with the constant-shape extrapolation described in
[`_grid_corners`](@ref).
"""
@inline function _grid_corners_dw(model::GriddedPSFModel{T}, Y, X) where {T}
    if length(model.psfs) == 1
        z = zero(T)
        return (1, one(T), z, z), (0, z, z, z), (0, z, z, z), (0, z, z, z)
    end
    ygrid, xgrid = model.ygrid, model.xgrid
    ny, nx = length(ygrid), length(xgrid)
    iy = clamp(searchsortedlast(ygrid, Y), 1, ny - 1)
    ix = clamp(searchsortedlast(xgrid, X), 1, nx - 1)
    y0, y1 = ygrid[iy], ygrid[iy + 1]
    x0, x1 = xgrid[ix], xgrid[ix + 1]
    # Clamp only the weight-defining position; the true (Y, X) is used
    # unclamped everywhere else (e.g. when recentering node PSFs).
    Yc = clamp(Y, y0, y1)
    Xc = clamp(X, x0, x1)
    cy = (Y > y0 && Y < y1) ? one(T) : zero(T) # dYc/dY; zero when clamped
    cx = (X > x0 && X < x1) ? one(T) : zero(T) # dXc/dX; zero when clamped
    invnorm = inv((x1 - x0) * (y1 - y0))
    dy1 = y1 - Yc
    dy0 = Yc - y0
    dx1 = x1 - Xc
    dx0 = Xc - x0
    w_ll, w_lr, w_ul, w_ur = dx1 * dy1 * invnorm, dx0 * dy1 * invnorm, dx1 * dy0 * invnorm, dx0 * dy0 * invnorm
    dll = (-dx1 * cy * invnorm, -dy1 * cx * invnorm)
    dlr = (-dx0 * cy * invnorm, dy1 * cx * invnorm)
    dul = (dx1 * cy * invnorm, -dy0 * cx * invnorm)
    dur = (dx0 * cy * invnorm, dy0 * cx * invnorm)
    ig = model.index_grid
    return (
        (ig[iy, ix], w_ll, dll[1], dll[2]), (ig[iy, ix + 1], w_lr, dlr[1], dlr[2]),
        (ig[iy + 1, ix], w_ul, dul[1], dul[2]), (ig[iy + 1, ix + 1], w_ur, dur[1], dur[2]),
    )
end

"""
    _grid_corners(model::GriddedPSFModel{T}, Y, X) -> NTuple{4, Tuple{Int, T}}

Return the (up to four) `(node_index, weight)` pairs of the node PSFs
bounding the query position `(Y, X)`, in (lower-left, lower-right,
upper-left, upper-right) order, following the bilinear-interpolation scheme
of `photutils.psf.GriddedPSFModel`. Inactive slots (only possible when
`length(model.psfs) == 1`) have `weight == 0` and must be skipped by the
caller rather than used to index `model.psfs`.

Positions outside the overall node grid are clamped into the nearest edge
cell *only for the purpose of computing weights*; this freezes the blended
*shape* at the grid boundary while still allowing the model to be evaluated
(recentered) at any true `(Y, X)`.
"""
@inline function _grid_corners(model::GriddedPSFModel, Y, X)
    corners = _grid_corners_dw(model, Y, X)
    return map(c -> (c[1], c[2]), corners)
end

"""
    _recenter_unit(node::AbstractPSFModel{T}, Y, X) where {T}

Return a copy of `node` recentered to `(Y, X)` with unit flux and zero
background, for blending into a [`GriddedPSFModel`](@ref) evaluation. The
node's other (shape) parameters are left unchanged.
"""
@inline function _recenter_unit(node::AbstractPSFModel{T}, Y, X) where {T}
    return ConstructionBase.setproperties(node, (y = Y, x = X, flux = one(T), bkg = zero(T)))
end

function evaluate(model::GriddedPSFModel{T}, py, px) where {T}
    Y, X = model.y, model.x
    s = zero(T)
    for (idx, w) in _grid_corners(model, Y, X)
        w == zero(T) && continue
        node = _recenter_unit(model.psfs[idx], Y, X)
        s = muladd(w, evaluate(node, py, px), s)
    end
    return muladd(model.flux, s, model.bkg)
end

function evaluate_fg(model::GriddedPSFModel{T}, py, px) where {T}
    Y, X = model.y, model.x
    s = zero(T)
    dsdy = zero(T)
    dsdx = zero(T)
    for (idx, w, dwdy, dwdx) in _grid_corners_dw(model, Y, X)
        w == zero(T) && continue
        node = _recenter_unit(model.psfs[idx], Y, X)
        # The recentered node's own y,x are (Y, X), so the first two
        # entries of its gradient are dphi/dY and dphi/dX (per the package
        # convention that ConstructionBase properties/gradients are
        # ordered y, x, ..., flux, bkg, validated at construction time).
        phi, G = evaluate_fg(node, py, px)
        s = muladd(w, phi, s)
        dsdy = muladd(dwdy, phi, muladd(w, G[1], dsdy))
        dsdx = muladd(dwdx, phi, muladd(w, G[2], dsdx))
    end
    f = muladd(model.flux, s, model.bkg)
    return f, SA[model.flux * dsdy, model.flux * dsdx, s, one(T)]
end

"""
    extent(model::GriddedPSFModel, fwhm_factor=5)

Return the union of the (up to four) active corner nodes' own `extent`,
each recentered to `(model.y, model.x)`. This works uniformly whether the
active corners are `ImagePSF` (which has its own exact data-extent override)
or analytic models (which fall back to the generic FWHM-based `extent`).
"""
function extent(model::GriddedPSFModel{T}, fwhm_factor = 5) where {T}
    Y, X = model.y, model.x
    ymin = ymax = xmin = xmax = zero(T)
    first_corner = true
    for (idx, w) in _grid_corners(model, Y, X)
        w == zero(T) && continue
        node = _recenter_unit(model.psfs[idx], Y, X)
        (ylo, yhi), (xlo, xhi) = extent(node, fwhm_factor)
        if first_corner
            ymin, ymax, xmin, xmax = ylo, yhi, xlo, xhi
            first_corner = false
        else
            ymin, ymax = min(ymin, ylo), max(ymax, yhi)
            xmin, xmax = min(xmin, xlo), max(xmax, xhi)
        end
    end
    return (ymin, ymax), (xmin, xmax)
end
