```@meta
CurrentModule = CrowdPhot
```

# Centroid Refinement and Morphology

After matched-filter detection identifies candidate sources, two
post-detection steps are needed before PSF fitting: refining sub-pixel
centroids and measuring morphological diagnostics that distinguish
stellar sources from cosmic rays, hot pixels, and extended objects.

## Centroiding

CrowdPhot provides [`centroid_poly`](@ref), a fast polynomial centroiding algorithm
based on [Vakili2016](@citet).  It fits a quadratic 2-D
polynomial to the 3×3 patch surrounding the brightest pixel of a
PSF-correlated image.  Under the well-sampled,
approximately Gaussian assumptions studied by [Vakili2016](@citet), the polynomial
centroid can approach the Cramér-Rao lower bound and takes ``\sim\!200`` ns per
source.  Correlating the observed image with the PSF broadens
source profiles helps to improve centroid estimation for undersampled
images.

Both the polynomial centroid and inverse-variance-weighted
center-of-mass (COM) centroid are returned, along with their full
covariance matrices and 1-σ uncertainties.  The COM centroid is
effectively free -- all required moments are already computed during
the normal-equation assembly for the polynomial fit.

[`choose_centroid`](@ref) selects between the two estimators
automatically: the polynomial centroid is used for well-sampled data,
while the COM centroid is preferred when the polynomial's curvature
matrix is nearly singular (e.g. for very broad PSFs).

In addition to the centroid and its covariance,
[`centroid_poly`](@ref) also returns morphological diagnostics for the
core of the source within the central 3x3 pixel box at near-zero additional
cost (see [Morphological Measurements](@ref) below):

- `sharpness` -- negated Laplacian of the quadratic fit; positive for peaks,
  near zero for flat regions.  Useful for distinguishing stars from cosmic
  rays and hot pixels.
- `roundness1_core` -- DAOPHOT SROUND / photutils `roundness1` on the 3×3
  patch; 0 = symmetric.
- `roundness2_core` -- DAOPHOT GROUND / photutils `roundness2` from the
  quadratic fit curvatures; 0 = circular, negative = extended in x.

```@docs
centroid_poly
CrowdPhot._centroid_poly3
choose_centroid
```

## Morphological Measurements

[`measure_star_shape`](@ref) computes aperture-scale morphological
diagnostics from inverse-variance-weighted second moments over the full
cutout.  It complements the core diagnostics from
[`centroid_poly`](@ref) with measurements that are integrated over the
entire stellar profile rather than just the 3×3 core. Combining the core
and full-aperture diagnostics provides a multi-scale view of the source's
morphology.

### Roundness Conventions

CrowdPhot follows the DAOPHOT and photutils conventions for roundness,
with two complementary statistics:

| Statistic | CrowdPhot (core) | CrowdPhot (aperture) | photutils | DAOPHOT | Meaning |
|-----------|-----------------|---------------------|-----------|---------|---------|
| SROUND | `roundness1_core` | `roundness1_aperture` | `roundness1` | `ROUND` | Bilateral vs. fourfold symmetry; 0 = symmetric |
| GROUND | `roundness2_core` | `roundness2_aperture` | `roundness2` | `DY` | Marginal Gaussian height ratio; 0 = circular, negative = extended in x, positive = extended in y |

Both conventions use 0 as the value for an ideal stellar source, with
nonzero values indicating asymmetry (SROUND) or ellipticity (GROUND).

**SROUND** tests whether the light distribution has fourfold symmetry
(the property of a circular Gaussian).  It sums pixel values along
``\pm 45^\circ`` axes: ``\Sigma_2`` measures the bilateral asymmetry
and ``\Sigma_4`` provides fourfold normalization.  SROUND catches
one-sided features like diffraction spikes and cosmic ray trails that
GROUND may miss.

**GROUND** compares the heights of best-fit marginal Gaussians in x and
y.  For the quadratic fit in [`_centroid_poly3`](@ref), these heights
satisfy ``H_X \propto \sqrt{|d|}`` and ``H_Y \propto \sqrt{|f|}``, so
GROUND can be computed directly from the quadratic coefficients without
a separate marginal fit.  The aperture version derives the same
quantity from the second central moments of the full cutout.

### Sharpness

The `sharpness` field returned by [`centroid_poly`](@ref) is the
negated Laplacian ``-(2d+2f)`` of the quadratic fit -- the mean upward
curvature at the peak.  It is proportional to ``1/\mathrm{FWHM}^2`` for
a Gaussian, making it a fast discriminator: cosmic rays (single bright
pixels) produce values orders of magnitude larger than stellar PSFs of
the expected width.

DAOPHOT and photutils define sharpness as
``(D_\mathrm{center} - \bar{D}_\mathrm{surrounding}) / H``, where
``D`` is the raw image and ``H`` is the convolved brightness
enhancement.  That definition requires both the raw and convolved
images, so it belongs in the candidate-selection layer rather than in
[`centroid_poly`](@ref).  The curvature-based `sharpness` is a
zero-cost proxy available at centroiding time.

### Core vs. Aperture Measurements

| Aspect | Core (`centroid_poly`) | Aperture (`measure_star_shape`) |
|--------|----------------------|--------------------------------|
| Scale | 3×3 central patch | Full cutout |
| Cost | ~200 ns (free with centroid) | ~1.5 μs (21×21) |
| Roundness accuracy | Limited by 3×3 sampling | Integrates over full profile |
| FWHM | Not available | Moment-based (Gaussian approx.) |
| Best use | Fast pre-filter at detection time | Candidate evaluation before PSF fitting |

```@docs
measure_star_shape
```

## References
This page cites the following references:

```@bibliography
Pages = ["morphology.md"]
Canonical = false
```
