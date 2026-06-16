```@meta
CurrentModule = CrowdPhot
```

# PSF Fitting Photometry

[`fit_all_stars`](@ref) performs PSF-fitting photometry on all sources in an
image using a DOLPHOT-style multi-pass algorithm. Stars are sorted by
brightness and fitted sequentially against a progressive residual image:
each fitted model is subtracted before the next star is processed, so
fainter neighbors are measured after brighter stars have been removed. On
subsequent passes each star is added back, re-fitted, and re-subtracted,
refining all measurements iteratively.

This avoids the large matrix solves used by fully simultaneous
group-fitting methods such as DAOPHOT/ALLSTAR. For a group containing `N`
stars, the cost of forming normal matrices can scale like `N^2`,
while a dense least-squares solve can scale like `N^3` in the group size.
In contrast, when the fitting radius, the residual-image approach scales
approximately linearly with the number of stars and passes:
``N_passes * N_stars``. By operating on the progressive residual image,
this method can give good results even in crowded fields.

## Result type

```@docs
MultiPassPhotResult
```

## Fitting function

```@docs
fit_all_stars
```
