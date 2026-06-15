```@meta
CurrentModule = CrowdPhot
```

# PSF Fitting Photometry

[`fit_all_stars`](@ref) performs PSF-fitting photometry on all sources in an
image using a DOLPHOT-style multi-pass algorithm.  Stars are sorted by
brightness and fitted sequentially against a progressive residual image:
each fitted model is subtracted before the next star is processed, so
fainter neighbours are measured after brighter stars have been removed.  On
subsequent passes each star is added back, re-fitted, and re-subtracted,
refining all measurements iteratively.

This approach is closely related to the DAOPHOT `ALLSTAR` task
([Stetson1987](@citet)) and the HST DOLPHOT package, which extend the idea
to multiple complete passes over the image.  The key insight is that
by photometering on a residual image with all other sources removed,
blending-induced biases are substantially reduced.

## Result type

```@docs
MultiPassPhotResult
```

## Fitting function

```@docs
fit_all_stars
```
