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

## Finalize step for bright stars

The fitting radius `fit_rad` used by `fit_all_stars` is normally kept small
(e.g. a handful of pixels) since larger fitting regions increase the cost
of every LM solve. This is fine for typical stars, but has two downsides
for bright stars when the PSF model itself extends much further than
`fit_rad` — for example a tabulated `ImagePSF`/`GriddedPSFModel` built from
optical simulations (such as the CRDS reference files used for Roman) that
remain a good description of the PSF out to large radii:

- The small-region flux/flux-uncertainty estimate is unbiased but not
  optimal — real signal-to-noise available in the well-modeled wings is
  discarded.
- For very bright stars, only subtracting the small central region leaves
  most of the star's wing flux in the residual, where it can contaminate
  the photometry of fainter neighbors.

The `finalize_snr_min` and `finalize_rad` keyword arguments to
[`fit_all_stars`](@ref) address this. When a star's small-footprint SNR
(`flux / flux_err`, evaluated once from the pass-1 fit — this decision is
never revisited on later passes) meets or exceeds `finalize_snr_min`, its
flux and flux uncertainty are re-measured, on *every* pass, with a
closed-form weighted-least-squares sum (no LM iteration — position,
background, and shape stay fixed at their small-footprint fitted values)
evaluated over a larger footprint set by `finalize_rad` (or the model's
natural extent when `finalize_rad` is not given). That larger footprint,
not the small fitting footprint, is what gets subtracted from the residual,
so later stars — including ones processed within the very same pass — see
a wing-cleaned residual as early as possible.

`finalize_snr_min` is `Inf` by default, so this feature is opt-in. It is
most useful for saturated or very bright stars fit with a wide,
well-characterized PSF model such as [`roman_crds_gridded_epsf`](@ref); for
compact analytic models (Gaussian, Moffat, Airy) whose tails are formally
infinite, an explicit `finalize_rad` is recommended over relying on the
default natural extent (a `5xFWHM` box), since "the model's natural extent"
is a somewhat arbitrary notion for such models — `fit_all_stars` emits a
warning in this situation.

Goodness-of-fit diagnostics are scoped deliberately: `qfit`, `qfit_expected`,
and `qfit_z` always stay computed over the small core fitting footprint,
regardless of finalize status, since an aggregate computed over a much
larger footprint would be dominated by wing-noise pixels carrying no
information about core model mismatch — diluting (or, for the
noise-standardized `qfit_z`, reducing statistical power to detect) exactly
the defects these diagnostics exist to catch. `crowding`, by contrast, is
recomputed over whichever footprint actually produced the reported flux,
since it is a statement about that specific measurement rather than a
general fit-quality diagnostic.

## Result type

```@docs
MultiPassPhotResult
```

## Fitting function

```@docs
fit_all_stars
```
