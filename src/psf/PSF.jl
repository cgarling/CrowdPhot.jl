module PSF

import ..CrowdPhot: AbstractLMDamping, AbstractScaleEstimator, AbstractCovarianceEstimator, LMResult, MarquardtDamping, MADScale, FixedScale, MScale, estimate_scale, TukeyLoss, weight, KnownWeightsCovarianceEstimator, ReweightedCovarianceEstimator, LMProblem, lm_irls
using ..CrowdPhot: sigma_clip, sigma_clip!, besselj0, besselj1, _clamp_inds
import ConstructionBase
import LossFunctions
import LoopVectorization as LV
using SpecialFunctions: erf
using StaticArrays: SA, SVector, MMatrix
using Statistics: median, mean, quantile, std

export AbstractPSFModel, AiryPSF, CircularGaussianPSF, GaussianPSF, CircularGaussianPRF, GaussianPRF, CircularMoffatPSF, MoffatPSF, ImagePSF, GriddedPSFModel, roman_crds_gridded_epsf
export evaluate, evaluate_fg, centroid, integral, render, peak, amplitude, effective_area, fit_star, fit_psf
export LMResult, MADScale, FixedScale, MScale, estimate_scale, TukeyLoss, weight, KnownWeightsCovarianceEstimator, ReweightedCovarianceEstimator

"""AbstractPSFModel{T}: Abstract type for PSF models with element type `T`. All PSF models should be subtypes of this abstract type, and implement the following methods:"""
abstract type AbstractPSFModel{T} end
Base.eltype(::AbstractPSFModel{T}) where {T} = T
Base.Broadcast.broadcastable(m::AbstractPSFModel) = Ref(m)
(model::AbstractPSFModel)(y, x) = evaluate(model, y, x)
(model::AbstractPSFModel)(idx::CartesianIndex) = evaluate(model, idx)
evaluate(model::AbstractPSFModel, idx::CartesianIndex) = evaluate(model, idx[1], idx[2])
evaluate_fg(model::AbstractPSFModel, idx::CartesianIndex) = evaluate_fg(model, idx[1], idx[2])
evaluate_fgh(model::AbstractPSFModel, idx::CartesianIndex) = evaluate_fgh(model, idx[1], idx[2])
function evaluate_fg(model::AbstractPSFModel, y, x, free_idx::AbstractVector)
    f, g = evaluate_fg(model, y, x)
    return f, view(g, free_idx)
end
function evaluate_fg(model::AbstractPSFModel, y, x, free_idx::SVector)
    f, g = evaluate_fg(model, y, x)
    return f, g[free_idx]
end

"""
    evaluate(model::AbstractPSFModel{T}, y::Real, x::Real)::T

Evaluate the PSF model at position `(y, x)`, where `y` is the row
(first array index) and `x` is the column (second array index).
"""
function evaluate end

function Base.convert(to::Type{T}, from::AbstractPSFModel{S}) where 
    {T1, T <: AbstractPSFModel{T1}, S}
    T === typeof(from) && return from
    ConstructionBase.constructorof(T) === ConstructionBase.constructorof(typeof(from)) ||
        throw(MethodError(convert, (to, from)))
    props = map(x -> T1(x), ConstructionBase.getproperties(from))
    return ConstructionBase.constructorof(T)(; props...)
end

# Declare that evaluate is allowed to be used inside @turbo loops.
# This will be true for *all* evaluate methods defined, so this is
# a contract that all PSF models must implement evaluate in a way that is compatible with @turbo.
LV.can_turbo(::typeof(evaluate), ::Val{3}) = true

# Workaround for a gap in VectorizationBase's `vcopysign` method table.
# When `@turbo` unrolls a loop dimension *without* vectorizing it (common on
# narrow 128-bit registers, e.g. Apple silicon, where `pick_vector_width(Float64)`
# is only 2), operands that depend solely on that dimension arrive as
# `VecUnroll{N,1,T,T}`, i.e. a tuple of plain scalars rather than of `Vec`s.
# `Base.copysign(::VecUnroll, ::VecUnroll)` then `fmap`s down to
# `vcopysign(::T, ::T)` for `T <: Base.HWReal`, which VectorizationBase does not
# define (it only covers `Vec`/`MM`/`VecUnroll` arguments). That MethodError is
# hit by `SpecialFunctions.erf` -> `VectorizationBase.verf`, which `copysign`s
# its result, so every `@turbo` loop over `CircularGaussianPRF`/`GaussianPRF`
# (and `GriddedPSFModel`s built from them) fails on such machines.
# The scalar fallback is exactly `Base.copysign`.
LV.VectorizationBase.vcopysign(a::Base.HWReal, b::Base.HWReal) = Base.copysign(a, b)

"""
    _turbo_safe(model) -> Bool
    _turbo_safe(::Type{<:AbstractPSFModel}) -> Bool

Whether a plain two-dimensional `for j in xr, i in yr; ... evaluate(model, i, j)`
loop over `model` may be wrapped in `LV.@turbo`.  Defaults to `true`; the
gathering models opt out.

`ImagePSF` (and `GriddedPSFModel`s built from it) must opt out because the
branch-free `ifelse` guards in `bicubic_interpolate` are handed a *doubly*
nested `VecUnroll` mask whenever LoopVectorization unrolls both loop
dimensions, and VectorizationBase defines no `ifelse` for a nested-`VecUnroll`
mask.  Double unrolling happens once the register width is small enough
(`pick_vector_width(Float64) == 2` on 128-bit NEON, i.e. Apple silicon), so
these loops throw a `MethodError` there while working on wider hardware.
Since these models gather per pixel anyway, the `@inbounds @simd` fallback is
used unconditionally rather than being gated on the host register size.
"""
_turbo_safe(::Type{<:AbstractPSFModel}) = true
_turbo_safe(model::AbstractPSFModel) = _turbo_safe(typeof(model))

"""
    centroid(model::AbstractPSFModel{T}) → (y::T, x::T)

Return the centroid of the PSF model as a tuple `(y, x)`, where `y` is the
row coordinate and `x` is the column coordinate;
default implementation assumes the centroid is given by fields `x` and `y` in the model struct.
"""
function centroid(model::AbstractPSFModel)
    if hasproperty(model, :x) && hasproperty(model, :y)
        return (model.y, model.x)
    else
        error("Model does not have `x` and `y` fields; either add them or implement `centroid(model)` for this model type.")
    end
end

"""
    integral(model::AbstractPSFModel{T})::T

Return the integral of the PSF model over all space;
default implementation assumes the integral is given by a field `flux` in the model struct.
"""
function integral(model::AbstractPSFModel)
    if hasproperty(model, :flux)
        return model.flux
    else
        error("Model does not have a `flux` field; either add one or implement `integral(model)` for this model type.")
    end
end

"""
    background(model::AbstractPSFModel{T})::T
Return the background level of the PSF model; 
if `bkg` field exists, return that, otherwise return `zero(T)`.
"""
function background(model::AbstractPSFModel)
    if hasproperty(model, :bkg)
        return model.bkg
    else
        error("Model does not have a `bkg` field; either add one or implement `background(model)` for this model type.")
    end
end

"""
    peak(model::AbstractPSFModel{T})::T

Return the peak value of the PSF model. By default, this
function evaluates the model at its centroid, but
models can override this.
"""
peak(model::AbstractPSFModel) = evaluate(model, centroid(model)...)

"""
    amplitude(model::AbstractPSFModel{T})::T

Return the amplitude of the PSF model, defined as the peak value
minus the background. Default implementation is
`peak(model) - background(model)`.
"""
amplitude(model::AbstractPSFModel) = peak(model) - background(model)

@doc raw"""
    effective_area(model::AbstractPSFModel{T})::T

Return the effective area of the PSF model, defined as

```math
\frac{\left(\int PSF(x, y) \, dx \, dy\right)^2}{\int PSF(x, y)^2 \, dx \, dy}
```

For PSF fitting 
photometry, this is the effective number of noisy pixels that contribute
to the measurement -- the variance of the flux measurement is approximately
the variance of the background noise per pixel times the effective area. 
Models with more complex definitions of effective area should implement 
their own version of this function.

For a star image with PSF `P` and background noise per pixel with variance `σ²`, 
the maximum likelihood estimate of the flux is

```math
\hat{F} = \frac{\sum_i P_i (D_i - B_i)}{\sum_i P_i^2}
```

where `D_i` is the observed data, `B_i` is the background, and `P_i` is the PSF
value at pixel `i`. The variance of the flux measurement is `var(\hat{F}) = σ² * effective_area(P)`.
"""
function effective_area(model::AbstractPSFModel) end

"""
    fwhm(model::AbstractPSFModel{T}) → (y_fwhm::T, x_fwhm::T)

Return the full width at half maximum (FWHM) of the PSF model as a tuple `(y_fwhm, x_fwhm)` in the y (row) and x (column) directions. By default, this function checks for a single `fwhm` field and returns it for both axes, or separate `x_fwhm` and `y_fwhm` fields if they exist. Models with more complex definitions of FWHM should implement their own version of this function.
"""
function fwhm(model::AbstractPSFModel)
    if hasproperty(model, :fwhm)
        return (model.fwhm, model.fwhm)
    elseif hasproperty(model, :x_fwhm) && hasproperty(model, :y_fwhm)
        return (model.y_fwhm, model.x_fwhm)
    else
        error("Model does not have `fwhm` or `x_fwhm` and `y_fwhm` fields; either add them or implement `fwhm(model)` for this model type.")
    end
end

"""
    theta(model::AbstractPSFModel{T}) → θ::T

Return the rotation angle `theta` of the PSF model in degrees CCW from the x-axis. By default, this function checks for a `theta` field and returns it, or **returns zero if no such field exists**. Models with more complex definitions of rotation should implement their own version of this function.
"""
function theta(model::AbstractPSFModel{T}) where {T}
    if hasproperty(model, :theta)
        return model.theta
    else
        return zero(T)
    end
end

"""
    evaluate_fg(model::AbstractPSFModel{T}, y::Real, x::Real) → (f::T, G::SVector{T})

Returns the model value `f` and partial derivatives of the `model`
with respect to the parameters `G` at position `(y, x)`, where `y` is the
row coordinate and `x` is the column coordinate. `G` follows
`ConstructionBase.getproperties(model)` order; model-center derivatives use
`y` before `x`.
"""
function evaluate_fg end

"""
    evaluate_fgh(model::AbstractPSFModel{T}, y::Real, x::Real) → (f::T, G::SVector{T}, H::SMatrix{T})
Returns the model value `f`, partial derivatives `G`, and Hessian matrix `H` of the `model`
with respect to the parameters at position `(y, x)`, where `y` is the row coordinate and
`x` is the column coordinate. `G` and `H` follow
`ConstructionBase.getproperties(model)` order; model-center derivatives use
`y` before `x`.
"""
function evaluate_fgh end

"""
    ellipse_bounds(a, b, θ) → (x_bound, y_bound)

Returns the half-width and half-height of the smallest axis-aligned
rectangle enclosing an ellipse with semi-major axis `a`, semi-minor
axis `b`, and rotation angle `θ` (degrees counterclockwise from the
x-axis).

```jldoctest
julia> using CrowdPhot.PSF: ellipse_bounds

julia> ellipse_bounds(3, 2, 0)
(3.0, 2.0)

julia> ellipse_bounds(3, 2, 90)
(2.0, 3.0)

julia> round.(ellipse_bounds(3, 2, 45); digits=3)
(2.55, 2.55)
```
"""
function ellipse_bounds(a, b, θ)
    θ = deg2rad(θ)
    sinθ, cosθ = sincos(θ)
    x_bound = hypot(a * cosθ, b * sinθ)
    y_bound = hypot(a * sinθ, b * cosθ)
    return x_bound, y_bound
end

"""
    extent([T::Integer], model::AbstractPSFModel, 
        fwhm_factor=5; roundint::Bool=false) → (y_range::Tuple, x_range::Tuple)

Returns the extent of the PSF model which is typically useful for fitting, plotting, etc., 
`((y_min, y_max), (x_min, x_max))`, where the first tuple is the row (y) range and the
second is the column (x) range. By default, the extent is the smallest axis-aligned
rectangle enclosing an ellipse centered on the model centroid whose major and minor axis
lengths are `fwhm_factor` times the model FWHM values. Models with more
complex shapes can override this function to provide a more appropriate extent.

If the first argument is an integer type `T`, the returned extent will be rounded to
the nearest integers of type `T` that fully contain the original extent. This is useful
for determining pixel indices for rendering and fitting.

```jldoctest
julia> using CrowdPhot.PSF: extent, CircularGaussianPSF, GaussianPSF

julia> extent(CircularGaussianPSF(y=20, x=10, fwhm=5, flux=30, bkg=1), 5)
((7.5, 32.5), (-2.5, 22.5))

julia> extent(GaussianPSF(y=20, x=10, x_fwhm=5, y_fwhm=3, theta=90, flux=30, bkg=1), 5)
((7.5, 32.5), (2.5, 17.5))

julia> extent(Int, GaussianPSF(y=20, x=10, x_fwhm=5, y_fwhm=3, theta=90, flux=30, bkg=1), 5)
((7, 33), (2, 18))
```
"""
function extent(model::AbstractPSFModel, fwhm_factor = 5)
    # default extent is 5x5 fwhm around centroid, but specific models can override this
    y0, x0 = centroid(model)
    FWHM = fwhm(model)
    a, b = fwhm_factor * FWHM[2] / 2, fwhm_factor * FWHM[1] / 2
    dx, dy = ellipse_bounds(a, b, theta(model))
    return (y0 - dy, y0 + dy), (x0 - dx, x0 + dx)
end
@inline function extent(::Type{T}, model::AbstractPSFModel, fwhm_factor = 5) where {T <: Integer}
    (ymin, ymax), (xmin, xmax) = extent(model, fwhm_factor)
    return (floor(T, ymin), ceil(T, ymax)), (floor(T, xmin), ceil(T, xmax))
end

"""
    CartesianIndices(model::AbstractPSFModel, [fwhm_factor]) -> CartesianIndices{2}

Return the `CartesianIndices` covering the integer-rounded `extent` of `model`.
Each `CartesianIndex` `idx` satisfies `idx[1] = y` (row) and `idx[2] = x`
(column), consistent with `image[y, x]` indexing.
"""
function Base.CartesianIndices(model::AbstractPSFModel, fwhm_factor = 5)
    ex = extent(Int, model, fwhm_factor)
    return CartesianIndices((ex[1][1]:ex[1][2], ex[2][1]:ex[2][2]))
end

"""
    render(model::AbstractPSFModel{T})::Matrix{T} where {T}

Return an **odd-sized** matrix covering the region returned by `extent(model)`.
The matrix is centered on the rounded model centroid; the half-width in each
dimension is chosen so that the full extent is covered and the total size is
odd (required for use as a correlation kernel in [`CrowdPhot.correlate`](@ref)).
Dimension 1 (rows) corresponds to the y (row) coordinate and dimension 2 (columns)
corresponds to the x (column) coordinate, consistent with `image[y, x]` indexing.
"""
function render(model::AbstractPSFModel{T}) where T
    (y_lo, y_hi), (x_lo, x_hi) = extent(model)
    y0, x0 = centroid(model)
    # Half-width large enough to cover the extent on both sides of the centroid.
    hx = max(ceil(Int, x0 - x_lo), ceil(Int, x_hi - x0))
    hy = max(ceil(Int, y0 - y_lo), ceil(Int, y_hi - y0))
    # Center on the nearest pixel to the true centroid.
    xc = round(Int, x0)
    yc = round(Int, y0)
    # Number of pixels along each dimension
    nx = 2hx + 1
    ny = 2hy + 1
    result = Matrix{T}(undef, ny, nx)
    if _turbo_safe(model)
        LV.@turbo for j in 1:nx
            xi = xc - hx + j - 1
            for i in 1:ny
                yi = yc - hy + i - 1
                result[i, j] = evaluate(model, yi, xi)
            end
        end
    else
        @inbounds for j in 1:nx
            xi = xc - hx + j - 1
            @simd for i in 1:ny
                yi = yc - hy + i - 1
                result[i, j] = evaluate(model, yi, xi)
            end
        end
    end
    return result
end

"""
    add_star!(out::AbstractMatrix, model::AbstractPSFModel,
              yr::AbstractUnitRange{<:Integer}, xr::AbstractUnitRange{<:Integer})

Mutate `out` by evaluating `model` at each pixel `(y, x)` for `y in yr, x in
xr` and adding the result to `out`, clamping `yr`/`xr` to the bounds of `out`
first (so pixels that lie off the edge of `out` are quietly skipped). This is
designed for rendering a PSF model into a larger image frame, requiring that
the `centroid` of the `model` be in the pixel space of the image (i.e., a
star with a center of `(y=20.5, x=10.5)` would be centered on the pixel at
row 20, column 10 of the image).

!!! note
    The SIMD-accelerated path is only used when `eltype(out)` matches the
    model's own element type `T` (`model::AbstractPSFModel{T}`) and falls back
    to a plain scalar loop otherwise.

See also: [`subtract_star!`](@ref) for the subtractive counterpart.
"""
function add_star!(out::AbstractMatrix{T}, model::AbstractPSFModel{T}, yr::AbstractUnitRange{<:Integer}, xr::AbstractUnitRange{<:Integer}) where {T}
    yr, xr = _clamp_inds(yr, xr, out)
    (isempty(yr) || isempty(xr)) && return out
    if _turbo_safe(model)
        LV.@turbo for j in xr
            for i in yr
                out[i, j] += evaluate(model, i, j)
            end
        end
    else
        @inbounds for j in xr
            @simd for i in yr
                out[i, j] += evaluate(model, i, j)
            end
        end
    end
    return out
end
function add_star!(out::AbstractMatrix, model::AbstractPSFModel, yr::AbstractUnitRange{<:Integer}, xr::AbstractUnitRange{<:Integer})
    yr, xr = _clamp_inds(yr, xr, out)
    (isempty(yr) || isempty(xr)) && return out
    @inbounds @simd for j in xr
        for i in yr
            out[i, j] += evaluate(model, i, j)
        end
    end
    return out
end
add_star!(out::AbstractMatrix, model::AbstractPSFModel, inds::CartesianIndices) = add_star!(out, model, inds.indices...)
function add_star!(out::AbstractMatrix, model::AbstractPSFModel)
    (y_lo, y_hi), (x_lo, x_hi) = extent(Int, model)
    return add_star!(out, model, y_lo:y_hi, x_lo:x_hi)
end


"""
    subtract_star!(out::AbstractMatrix, model::AbstractPSFModel,
        yr::AbstractUnitRange{<:Integer}, xr::AbstractUnitRange{<:Integer})

Subtract the model PSF flux from each pixel of `out` over `y in yr, x in
xr`, i.e. `out[y, x] -= evaluate(model, y, x)`.

!!! note
    The SIMD-accelerated path requires `eltype(out) == T` for
    `model::AbstractPSFModel{T}` and falls back to a plain scalar loop otherwise.

See also: [`add_star!`](@ref) for the additive counterpart.
"""
function subtract_star!(out::AbstractMatrix{T}, model::AbstractPSFModel{T}, yr::AbstractUnitRange{<:Integer}, xr::AbstractUnitRange{<:Integer}) where {T}
    yr, xr = _clamp_inds(yr, xr, out)
    (isempty(yr) || isempty(xr)) && return out
    if _turbo_safe(model)
        LV.@turbo for j in xr
            for i in yr
                out[i, j] -= evaluate(model, i, j)
            end
        end
    else
        @inbounds for j in xr
            @simd for i in yr
                out[i, j] -= evaluate(model, i, j)
            end
        end
    end
    return out
end
function subtract_star!(out::AbstractMatrix, model::AbstractPSFModel, yr::AbstractUnitRange{<:Integer}, xr::AbstractUnitRange{<:Integer})
    yr, xr = _clamp_inds(yr, xr, out)
    (isempty(yr) || isempty(xr)) && return out
    @inbounds for j in xr
        for i in yr
            out[i, j] -= evaluate(model, i, j)
        end
    end
    return out
end
subtract_star!(out::AbstractMatrix, model::AbstractPSFModel, inds::CartesianIndices) = subtract_star!(out, model, inds.indices...)
function subtract_star!(out::AbstractMatrix, model::AbstractPSFModel)
    (y_lo, y_hi), (x_lo, x_hi) = extent(Int, model)
    return subtract_star!(out, model, y_lo:y_hi, x_lo:x_hi)
end

"""
    pixel_response_kernel(n::Integer)

Returns a discretized flat "pixel response" kernel of full-width `n` (the
oversampling factor), height 1/n so it sums to 1. Reproduces
`astropy.convolution.Box2DKernel(width=n).array` exactly (verified for
n = 1..8): for even n this is *not* a naive n×n uniform box -- it's the exact
overlap integral of a continuous box of width n against each unit-width pixel
bin, giving an (n+1)x(n+1) kernel with tapered edge/corner weights.
"""
function pixel_response_kernel(n::Integer)
    sz = isodd(n) ? n : n + 1
    half = n / 2
    centers = (0:sz-1) .- (sz - 1) / 2
    w1d = [clamp(min(c + 0.5, half) - max(c - 0.5, -half), 0, Inf) / n for c in centers]
    return w1d * w1d'
end

include("parametric_models.jl")
include("empirical_models.jl")
include("empirical_builder.jl")
include("gridded_psf.jl")
include("roman_crds.jl")
include("psf_fitting.jl")
include("pick.jl")

end # module PSF
