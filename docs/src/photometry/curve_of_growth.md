```@meta
CurrentModule = CrowdPhot
```

# Curves of Growth

[`curve_of_growth`](@ref) measures cumulative flux within concentric circular
apertures centered on a source.  For an image, it performs aperture photometry;
for a PSF model, it uses either analytic integrals (for models with closed-form
radial profiles) or pixel integration.

The primary use cases are:

- **PSF encircled energy**: measure what fraction of a PSF's total flux falls
  within a given radius.  Normalize the curve to get the encircled-energy
  fraction as a function of radius.
- **Aperture photometry**: measure the cumulative flux as a function of
  aperture radius to determine the total source flux and its uncertainty.
- **Aperture correction**: extrapolate from a finite-radius aperture to total
  flux using a PSF model's curve of growth.

## Result type

```@docs
CurveOfGrowth
```

## Computing curves of growth

```@docs
curve_of_growth(::AbstractMatrix{T}, ::Real, ::Real, ::AbstractVector{<:Real}) where {T}
curve_of_growth(::CrowdPhot.PSF.AbstractPSFModel{T}, ::AbstractVector{<:Real}) where {T}
```

## Single-radius queries

```@docs
encircled_flux
radius_at_flux
```

## Normalization

```@docs
normalize(::CurveOfGrowth)
```

## Apertures

!!! warning "Internal API"
    The aperture types below are used internally by [`curve_of_growth`](@ref)
    and are **not yet part of the public API**.  Their names and interfaces
    may change in future releases.

```@docs
CircularAperture
ExactOverlap
CenterOverlap
WholePixelOverlap
SubpixelOverlap
```