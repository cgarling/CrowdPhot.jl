# Curves of growth and encircled flux via concentric circular apertures.
#
# Provides the image-based `curve_of_growth` workhorse plus model-based dispatch
# with analytic shortcuts for models that have closed-form radial integrals.

import ConstructionBase
using .PSF:
    AbstractPSFModel, CircularGaussianPSF, CircularMoffatPSF,
    evaluate, centroid, integral, render

# ==============================================================================
# Result type
# ==============================================================================

"""
    CurveOfGrowth{T}

Result of [`curve_of_growth`](@ref).  All vectors have the same length (the
number of input radii).

# Fields

- `radii::Vector{T}`: the aperture radii (same as the input).
- `flux::Vector{T}`: cumulative flux within each radius.
- `flux_err::Vector{T}`: propagated 1-sigma flux uncertainty (empty if no
  `inv_var` was provided).
- `area::Vector{T}`: sum of aperture weights over valid pixels at each radius
  (equals ``\\pi r^2`` when the aperture is fully inside the image with no
  masked pixels).
- `y::T`, `x::T`: the center position used for the apertures.
"""
Base.@kwdef struct CurveOfGrowth{T}
    radii::Vector{T}
    flux::Vector{T}
    flux_err::Vector{T}
    area::Vector{T}
    y::T
    x::T
end

# ==============================================================================
# curve_of_growth — image method (the workhorse)
# ==============================================================================

"""
    curve_of_growth(image::AbstractMatrix, y, x, radii; kws...) -> CurveOfGrowth

Compute the curve of growth at `(y, x)` using concentric circular apertures
with the given `radii`.

# Arguments

- `image::AbstractMatrix`: the background-subtracted image.
- `y`, `x`: the pixel coordinate of the source center (row, column).
- `radii::AbstractVector{<:Real}`: aperture radii, must be strictly positive
  and increasing.

# Keyword arguments

- `method = ExactOverlap()`: the aperture-weight method
  ([`ExactOverlap`](@ref), [`CenterOverlap`](@ref), etc.).
- `inv_var = nothing`: per-pixel inverse variance for error propagation.
  Must be the same shape as `image`.  Pixels with non-positive or non-finite
  `inv_var` are excluded from the flux sum and area.
- `background = zero(T)`: constant background to subtract from `image` before
  weighting.

# Returns

A [`CurveOfGrowth`](@ref) with cumulative `flux`, `area`, and optionally
`flux_err` at each radius.
"""
function curve_of_growth(
        image::AbstractMatrix{T},
        y::Real,
        x::Real,
        radii::AbstractVector{<:Real};
        method = ExactOverlap(),
        inv_var = nothing,
        background = zero(T),
    ) where {T}
    FT = float(T)
    _validate_radii(radii)
    n = length(radii)
    flux = Vector{FT}(undef, n)
    area = Vector{FT}(undef, n)
    flux_err = isnothing(inv_var) ? FT[] : Vector{FT}(undef, n)
    has_error = !isnothing(inv_var)
    if has_error && size(inv_var) != size(image)
        throw(ArgumentError("inv_var must be the same size as image"))
    end

    for (k, r) in enumerate(radii)
        ap = CircularAperture(y, x, r)
        yr, xr = clipped_axes(ap, image)
        total_flux = zero(FT)
        total_area = zero(FT)
        total_var = zero(FT)
        for j in xr, i in yr
            w = aperture_weight(ap, i, j, method)
            w > 0 || continue
            total_flux += w * (FT(image[i, j]) - FT(background))
            total_area += w
            if has_error
                iv = inv_var[i, j]
                if isfinite(iv) && iv > 0
                    total_var += w^2 / iv
                end
            end
        end
        flux[k] = total_flux
        area[k] = total_area
        if has_error
            flux_err[k] = sqrt(total_var)
        end
    end

    return CurveOfGrowth{FT}(
        Vector{FT}(radii), flux, flux_err, area, FT(y), FT(x))
end

# ==============================================================================
# curve_of_growth — model method (generic pixel-integration fallback)
# ==============================================================================

"""
    curve_of_growth(model::AbstractPSFModel, radii; kws...) -> CurveOfGrowth

Compute the curve of growth for a PSF model.

The generic method evaluates `model` at each pixel inside the aperture
bounding box.  Models with analytic radial profiles
([`CircularGaussianPSF`](@ref), [`CircularMoffatPSF`](@ref)) override this
with closed-form integrals.

# Keyword arguments

- `method = ExactOverlap()`: the aperture-weight method.

See also [`encircled_flux`](@ref) for querying a single radius without
materializing the full curve.
"""
function curve_of_growth(
        model::AbstractPSFModel{T},
        radii::AbstractVector{<:Real};
        method = ExactOverlap(),
    ) where {T}
    _validate_radii(radii)
    FT = float(T)
    n = length(radii)
    y, x = centroid(model)
    FT_y, FT_x = FT(y), FT(x)
    flux = Vector{FT}(undef, n)
    area = Vector{FT}(undef, n)

    for (k, r) in enumerate(radii)
        ap = CircularAperture(y, x, r)
        yr, xr = bounding_axes(ap)
        total_flux = zero(FT)
        total_area = zero(FT)
        for j in xr, i in yr
            w = aperture_weight(ap, i, j, method)
            w > 0 || continue
            total_flux += w * FT(evaluate(model, i, j))
            total_area += w
        end
        flux[k] = total_flux
        area[k] = total_area
    end

    return CurveOfGrowth{FT}(Vector{FT}(radii), flux, FT[], area, FT_y, FT_x)
end

# ==============================================================================
# Analytic overloads for parametric models
# ==============================================================================

# CircularGaussianPSF: enclosed flux = flux * (1 - exp(γ * r² / fwhm²))
# where γ = GAUSS_PRE = -4*log(2).
function curve_of_growth(
        model::CircularGaussianPSF{T},
        radii::AbstractVector{<:Real};
        kws...
    ) where {T}
    _validate_radii(radii)
    FT = float(T)
    y, x = centroid(model)
    r = FT.(radii)
    γ = FT(PSF.GAUSS_PRE)               # -4*log(2)
    fwhm² = FT(model.fwhm)^2
    total = FT(integral(model))
    flux = total .* (1 .- exp.(γ .* r.^2 ./ fwhm²))
    return CurveOfGrowth{FT}(collect(r), flux, FT[], FT(π) .* r.^2, FT(y), FT(x))
end

# CircularMoffatPSF: enclosed flux = flux * (1 - (1 + r²/α²)^(1-β))
function curve_of_growth(
        model::CircularMoffatPSF{T},
        radii::AbstractVector{<:Real};
        kws...
    ) where {T}
    _validate_radii(radii)
    FT = float(T)
    y, x = centroid(model)
    r = FT.(radii)
    α² = FT(model.α)^2
    β  = FT(model.β)
    total = FT(integral(model))
    flux = total .* (1 .- (1 .+ r.^2 ./ α²).^(1 - β))
    return CurveOfGrowth{FT}(collect(r), flux, FT[], FT(π) .* r.^2, FT(y), FT(x))
end

# ==============================================================================
# encircled_flux — single-radius query
# ==============================================================================

"""
    encircled_flux(cog::CurveOfGrowth, r::Real) -> T

Return the encircled flux at radius `r` by linear interpolation of the
`cog` data.  Returns `NaN` for radii outside the range of the sampled radii.

See also [`curve_of_growth`](@ref), [`radius_at_flux`](@ref).
"""
function encircled_flux(cog::CurveOfGrowth{T}, r::Real) where {T}
    FT = float(T)
    rr = FT(r)
    # Trim to the monotonically-increasing prefix of the profile.
    radii, flux = _monotonic_prefix(cog.radii, cog.flux)
    if rr < first(radii) || rr > last(radii)
        return convert(FT, NaN)
    end
    # Linear interpolation (can be upgraded to cubic spline later).
    idx = searchsortedlast(radii, rr)
    idx == 0 && return flux[1]
    idx == length(radii) && return flux[end]
    t = (rr - radii[idx]) / (radii[idx+1] - radii[idx])
    return muladd(t, flux[idx+1] - flux[idx], flux[idx])
end

"""
    encircled_flux(model::AbstractPSFModel, r::Real; method=ExactOverlap()) -> T

Compute the encircled flux within radius `r` for a PSF model directly,
without materializing the full curve of growth.

Models with analytic radial profiles override this with closed-form
formulas.  The generic method integrates over aperture pixels.
"""
function encircled_flux(
        model::AbstractPSFModel{T},
        r::Real;
        method = ExactOverlap(),
    ) where {T}
    FT = float(T)
    y, x = centroid(model)
    ap = CircularAperture(y, x, r)
    yr, xr = bounding_axes(ap)
    total = zero(FT)
    for j in xr, i in yr
        w = aperture_weight(ap, i, j, method)
        w > 0 || continue
        total += w * FT(evaluate(model, i, j))
    end
    return total
end

# Analytic specializations for encircled_flux
function encircled_flux(
        model::CircularGaussianPSF{T},
        r::Real;
        kws...
    ) where {T}
    FT = float(T)
    rr = FT(r)
    γ = FT(PSF.GAUSS_PRE)
    fwhm² = FT(model.fwhm)^2
    return FT(integral(model)) * (1 - exp(γ * rr^2 / fwhm²))
end

function encircled_flux(
        model::CircularMoffatPSF{T},
        r::Real;
        kws...
    ) where {T}
    FT = float(T)
    rr = FT(r)
    α² = FT(model.α)^2
    β  = FT(model.β)
    return FT(integral(model)) * (1 - (1 + rr^2 / α²)^(1 - β))
end

# ==============================================================================
# radius_at_flux — inverse query
# ==============================================================================

"""
    radius_at_flux(cog::CurveOfGrowth, target_flux::Real) -> T

Return the radius enclosing `target_flux` by linear interpolation of the
monotonically-increasing prefix of the `cog` data.  Returns `NaN` if
`target_flux` lies outside the range of the profile.

See also [`encircled_flux`](@ref).
"""
function radius_at_flux(cog::CurveOfGrowth{T}, target_flux::Real) where {T}
    FT = float(T)
    ff = FT(target_flux)
    radii, flux = _monotonic_prefix(cog.radii, cog.flux)
    if ff < first(flux) || ff > last(flux)
        return convert(FT, NaN)
    end
    idx = searchsortedlast(flux, ff)
    idx == 0 && return radii[1]
    idx == length(flux) && return radii[end]
    t = (ff - flux[idx]) / (flux[idx+1] - flux[idx])
    return muladd(t, radii[idx+1] - radii[idx], radii[idx])
end

# ==============================================================================
# normalize
# ==============================================================================

"""
    normalize(cog::CurveOfGrowth; method::Symbol = :max) -> CurveOfGrowth

Return a normalized copy of `cog`.

# Keyword arguments

- `method = :max`: normalization strategy.  `:max` normalizes such that the
  maximum `flux` value is 1.  `:sum` normalizes such that the largest-radius
  flux value is 1.
"""
function normalize(cog::CurveOfGrowth{T}; method::Symbol = :max) where {T}
    norm = if method == :max
        maximum(cog.flux)
    elseif method == :sum
        cog.flux[end]
    else
        throw(ArgumentError("method must be :max or :sum, got $method"))
    end
    if norm == 0 || !isfinite(norm)
        throw(ArgumentError(
            "cannot normalize: $(method) value is zero or non-finite"))
    end
    return ConstructionBase.setproperties(cog, (;
        flux = cog.flux ./ norm,
        flux_err = cog.flux_err ./ norm,
    ))
end

# ==============================================================================
# Internal helpers
# ==============================================================================

"""
    _validate_radii(radii)

Check that `radii` is non-empty, strictly positive, and strictly increasing.
Throws `ArgumentError` otherwise.
"""
function _validate_radii(radii::AbstractVector{<:Real})
    n = length(radii)
    n > 0 || throw(ArgumentError("radii must be non-empty"))
    all(r -> r > 0, radii) ||
        throw(ArgumentError("radii must be strictly positive"))
    all(i -> radii[i] < radii[i+1], 1:n-1) ||
        throw(ArgumentError("radii must be strictly increasing"))
    return nothing
end

"""
    _monotonic_prefix(x, y) -> (x[1:k], y[1:k])

Trim `x` and `y` to the longest monotonically-increasing prefix of `y`.
Returns `(x, y)` unchanged if `y` is already monotonic.

This is used by the interpolation functions, which require a monotonic
profile for invertibility.
"""
function _monotonic_prefix(x::AbstractVector, y::AbstractVector)
    # Find the first index where y stops increasing.
    k = length(y)
    for i in 1:(length(y) - 1)
        if y[i+1] <= y[i]
            k = i
            break
        end
    end
    return x[1:k], y[1:k]
end

# ==============================================================================
# Reference encircled-energy curves
# ==============================================================================

"""
    reference_cog(instrument::Symbol, filter::Symbol) -> CurveOfGrowth

Load a reference encircled-energy curve for HST/ACS.

Data are from [Bohlin2016](@citet) for WFC and HRC, and [Avila (2016)](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/acs/documentation/instrument-science-reports-isrs/_documents/isr1605.pdf) for SBC,
sourced from the [STScI aperture-correction
page](https://www.stsci.edu/hst/instrumentation/acs/data-analysis/aperture-corrections).

# Available instruments and filters

| `instrument` | `filter` |
|:------------:|:---------|
| `:WFC` | `:F435W`, `:F475W`, `:F502N`, `:F555W`, `:F550M`, `:F606W`, `:F625W`, `:F658N`, `:F660N`, `:F775W`, `:F814W`, `:F892N`, `:F850LP` |
| `:HRC` | `:F220W`, `:F250W`, `:F330W`, `:F344N`, `:F435W`, `:F475W`, `:F502N`, `:F555W`, `:F550M`, `:F606W`, `:F625W`, `:F658N`, `:F660N`, `:F775W`, `:F814W`, `:F892N`, `:F850LP` |
| `:SBC` | `:F125LP`, `:F140LP`, `:F150LP` |

# Returns

A [`CurveOfGrowth`](@ref) whose `radii` are in detector pixels and whose
`flux` values are encircled-energy fractions (dimensionless, 0 to 1).  The
infinite-aperture endpoint (5.5″ for WFC/HRC, 4″ for SBC) is appended with
``\\mathrm{EE} = 1`` so that [`normalize`](@ref) and
[`encircled_flux`](@ref) work correctly across the full range.

`flux_err` is empty; `y` and `x` are zero.

# Examples

```julia
ref = reference_cog(:WFC, :F814W)
ee_at_3px = encircled_flux(ref, 3.0)
r80 = radius_at_flux(ref, 0.80)
```
"""
function reference_cog(instrument::Symbol, filter::Symbol)
    # Map instrument to file name, pixel scale (arcsec/pix), and
    # infinite-aperture radius in arcseconds.
    files = Dict(
        :WFC => ("bohlin2016_wfc_ee-1.txt", 0.05, 5.5),
        :HRC => ("bohlin2016_hrc_ee-1.txt", 0.025, 5.5),
        :SBC => ("avila2016_sbc_ee-1.txt", 0.03, 4.0),  # TODO: verify SBC pixel scale
    )
    entry = get(files, instrument) do
        throw(ArgumentError(
            "unknown instrument: $instrument. " *
            "Valid: $(join(keys(files), ", "))"))
    end
    fname, pix_scale, inf_radius_arcsec = entry
    data_dir = joinpath(dirname(@__DIR__), "data")
    radii, filters, ee_matrix = _parse_ee_file(joinpath(data_dir, fname))

    row = findfirst(==(string(filter)), filters)
    row === nothing && throw(ArgumentError(
        "filter $filter not found in $instrument table. " *
        "Available: $(join(filters, ", "))"))

    r = Float64.(radii)
    ee = Float64.(ee_matrix[row, :])

    # SBC table is in arcseconds; convert to pixels.
    if instrument == :SBC
        r ./= pix_scale
    end

    r_inf = inf_radius_arcsec / pix_scale

    # Trim radii beyond the infinite-aperture calibration radius.
    # These points have EE > 1.0 relative to the calibration
    # normalization — physically valid but outside the calibrated
    # magnitude system where EE(r_inf) ≡ 1.
    keep = r .<= r_inf
    r = r[keep]
    ee = ee[keep]

    # Append the infinite-aperture endpoint if not already present.
    if r[end] < r_inf
        r = [r; r_inf]
        ee = [ee; 1.0]
    end

    return CurveOfGrowth{Float64}(r, ee, Float64[], π .* r.^2, 0.0, 0.0)
end

function _parse_ee_file(path)
    lines = filter(readlines(path)) do l
        s = strip(l)
        return !isempty(s) && !startswith(s, "#")
    end
    isempty(lines) && error("empty reference file: $path")
    # Radii row starts with the -999.0 sentinel.
    radii = parse.(Float64, split(lines[1])[2:end])
    filters = String[]
    ee_rows = Float64[]
    for line in lines[2:end]
        parts = split(line)
        isempty(parts) && continue
        push!(filters, parts[1])
        append!(ee_rows, parse.(Float64, parts[2:end]))
    end
    n_filt = length(filters)
    n_rad = length(radii)
    ee_matrix = reshape(ee_rows, n_rad, n_filt)'
    return radii, filters, ee_matrix
end
