# Detection and Centroiding

## Centroiding

CrowdPhot provides a fast polynomial centroiding algorithm based on
[Vakili2016](@citet).  The method fits a quadratic 2-D
polynomial to the 3×3 patch surrounding the brightest pixel of a
PSF-correlated (matched-filtered) image.  The polynomial centroid
saturates the Cramér-Rao lower bound for well-sampled stars and is
``\sim\!200`` ns per source.

Both the polynomial centroid and an inverse-variance-weighted
center-of-mass (COM) centroid are returned, along with their full
covariance matrices and 1-σ uncertainties.  The COM centroid is
effectively free — all required moments are already computed during
the normal-equation assembly for the polynomial fit.

Both the polynomial centroid and an inverse-variance-weighted
center-of-mass (COM) centroid are returned, along with their full
covariance matrices and 1-σ uncertainties.  The COM centroid is
effectively free — all required moments are already computed during
the normal-equation assembly for the polynomial fit.

[`choose_centroid`](@ref) selects between the two estimators
automatically: the polynomial centroid is used for well-sampled data,
while the COM centroid is preferred when the polynomial's curvature
matrix is nearly singular (e.g. for very broad PSFs).

```@docs
centroid_poly
CrowdPhot._centroid_poly3
choose_centroid
```
