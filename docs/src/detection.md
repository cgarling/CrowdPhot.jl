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

```@docs
centroid_poly
CrowdPhot._centroid_poly3
```
