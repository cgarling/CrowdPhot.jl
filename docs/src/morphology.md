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
cost (see [Morphological Measurements](#Morphological-Measurements) below):

- `normalized_curvature` -- negated Laplacian divided by the fitted
  amplitude above `background`; ``\approx 16\log(2)/\mathrm{FWHM}^2`` for a
  circular Gaussian, flux-independent.
- `compactness_core` -- inverse of the total second central moment of the light
  distribution; larger for more compact light distributions.
- `roundness1_core` -- DAOPHOT SROUND / photutils `roundness1` on the 3×3
  patch; 0 = symmetric.
- `roundness2_core` -- DAOPHOT GROUND / photutils `roundness2` from the
  quadratic fit curvatures; 0 = circular, negative = extended in x.
- `ellipticity_core` -- rotationally invariant ellipticity of the
  fitted 3×3 core. Symmetric, round cores have `ellipticity_core ≈ 0`,
  while elongated cores have `ellipticity_core > 0`.

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
and ``\Sigma_4`` provides fourfold normalization.

**GROUND** compares the heights of best-fit marginal Gaussians in x and
y.  For the quadratic fit in [`_centroid_poly3`](@ref), these heights
satisfy ``H_X \propto \sqrt{|d|}`` and ``H_Y \propto \sqrt{|f|}``, so
GROUND can be computed directly from the quadratic coefficients without
a separate marginal fit.  The aperture version derives the same
quantity from the second central moments of the full cutout.

For elongation along a row or column, or a one-sided spike, both
statistics respond: the marginal second moments become unequal (GROUND)
and fourfold symmetry breaks (SROUND).  GROUND additionally reports the
*direction* of an axis-aligned stretch through its sign.

Their sensitivities differ for features at intermediate position
angles.  A source elongated at 45° keeps the marginal widths
``\sigma^2_{xx} \approx \sigma^2_{yy}`` equal, so GROUND stays near
zero, while the extra flux along one diagonal breaks fourfold symmetry
and drives SROUND strongly nonzero.  Isolated off-axis hot pixels
behave the same way: they inflate ``\sigma^2_{xx}`` and
``\sigma^2_{yy}`` about equally (GROUND near zero) but register clearly
in SROUND.  The rotationally invariant
`ellipticity_core` and `ellipticity_aperture` fields (below) measure
axis-ratio elongation at any position angle and close GROUND's 45° blind
spot directly. Measuring several statistics together lets us catch failure
modes that any one alone would miss.

Departures from a Gaussian that *preserve* fourfold symmetry,
e.g., a faint ring, leave all three of
SROUND, GROUND, and ellipticity near zero, so none of these core
statistics flags them. Such issues are better identified by a PSF-fit
residual.

### Ellipticity

`ellipticity_core` (from [`centroid_poly`](@ref)) and
`ellipticity_aperture` (from [`measure_star_shape`](@ref)) are
rotationally invariant shape statistics:

```math
\text{ellipticity} = 1 - \sqrt{\lambda_\mathrm{min} / \lambda_\mathrm{max}},
```

where ``\lambda_\mathrm{min} \le \lambda_\mathrm{max}`` are the
eigenvalues of a symmetric ``2\times2`` shape matrix.  For the core the
matrix is the negated Hessian ``-\!H = -\begin{bmatrix}2d & e\\ e & 2f\end{bmatrix}``
of the quadratic fit, whose eigenvalues are ``\propto`` the inverse
squared profile widths along the principal axes.  For the aperture it is
the second-moment covariance ``\begin{bmatrix}\sigma^2_{yy} & \sigma^2_{xy}\\ \sigma^2_{xy} & \sigma^2_{xx}\end{bmatrix}``.
In both cases ``\text{ellipticity} = 1 - b/a`` where ``b/a`` is the
minor-to-major axis ratio: ``0`` for a circular source, approaching
``1`` for a highly elongated one.

Because they use the full ``2\times2`` matrix (including the cross term
``e`` / ``\sigma^2_{xy}``), these fields detect elongation at any
position angle, unlike `roundness2_core` / `roundness2_aperture`, which
compare only axis-aligned marginals and read ``\approx 0`` for a source
elongated at 45°.  `ellipticity_core` returns `NaN` when the fitted core
is not a clean local maximum; `ellipticity_aperture` returns `NaN` when
the moment covariance is not positive definite.

### Normalized Curvature

The `normalized_curvature` field returned by [`centroid_poly`](@ref)
is the negated Laplacian of the quadratic fit divided by the fitted
amplitude above `background` ``-(2d+2f) / I_0``.  For a Gaussian this approximates
``16\log(2)/\mathrm{FWHM}^2`` and is independent of flux, making it a fast
discriminator: cosmic rays (single bright pixels) produce larger values
than stellar PSFs of the expected width. Saturated stars produce lower values
because their cores are flat.

### Compactness

`compactness_core` (from [`centroid_poly`](@ref)) and
`compactness_aperture` (from [`measure_star_shape`](@ref)) are the
inverse of the total weighted second central moment of the light
distribution:

```math
\text{compactness} = \frac{1}{\sigma_x^2 + \sigma_y^2},
```

where ``\sigma_x^2`` and ``\sigma_y^2`` are the inverse‑variance‑weighted
second central moments of the pixel values about the estimated centroid,
taken over the 3×3 core (`compactness_core`) or the full cutout
(`compactness_aperture`).  For a Gaussian profile this quantity is
proportional to the inverse of the FWHM squared, so larger values
indicate more compact (sharper) profiles.  Cosmic rays and hot pixels,
with negligible spatial extent, produce very large values.

!!! note "`compactness_core` needs the background removed"
    The compactness moments are weighted by the (background-subtracted)
    pixel values, so a sky pedestal that is *not* removed biases them.  A
    flat background has second central moment ``2/3`` per axis on the 3×3
    core, so an un‑subtracted `compactness_core` is a convex combination of
    the true source value and ``2/3``, weighted by the sky-to-source count
    ratio inside the box; for a faint source on a bright sky it collapses
    to ``\approx 0.75`` regardless of the true width.  `normalized_curvature`
    is likewise suppressed by an un‑subtracted pedestal through its
    division by the fitted peak, though its curvature numerator is
    pedestal-independent.  Pass the sky level via the `background` keyword
    of [`centroid_poly`](@ref) (or of [`measure_star_shapes`](@ref), which
    forwards it), or subtract it from the image beforehand.  The polynomial
    centroid itself is unaffected either way.  `compactness_aperture` is
    computed on ``\max(0,\,\text{image} - \text{background})`` and needs the
    same correct `background`.

`compactness_core` returns `NaN` when the weighted moment sum is
non‑positive (`R00 ≤ 0` or ``\sigma_x^2 + \sigma_y^2 \le 0``), which
happens for pure-noise peaks; `compactness_aperture` returns `NaN` under
the analogous degenerate conditions.  This lets non‑stellar detections
be rejected with a simple `isfinite` check.

### On sharpness

DAOPHOT and photutils define sharpness as
``(D_\mathrm{center} - \bar{D}_\mathrm{surrounding}) / H``, where
``D`` is the raw image and ``H`` is the convolved brightness
enhancement.  That definition requires both the raw and convolved
images, so it belongs in the candidate-selection layer rather than in
[`centroid_poly`](@ref). The `compactness_core` and `normalized_curvature` fields are complementary: they are available immediately during centroid refinement, require no convolution, and provide measures of compactness/sharpness to identify spurious and non‑stellar detections before fitting is performed.

### Core vs. Aperture Measurements

| Aspect | Core (`centroid_poly`) | Aperture (`measure_star_shape`) |
|--------|----------------------|--------------------------------|
| Scale | 3×3 central patch | Full cutout |
| Cost | ~200 ns (free with centroid) | ~1.5 μs (21×21), scales with pixel count |
| Roundness accuracy | Limited by 3×3 sampling | Integrates over full profile |
| FWHM | Not available (use `compactness_core`) | Marginal moment widths (Gaussian approx.) |
| Compactness / ellipticity | From the quadratic Hessian and 3×3 moments | From the full-profile moment covariance |
| Background | `background` kwarg (centroid unaffected; needed for the diagnostics) | `background` kwarg (subtracted, clamped at 0) |
| Best use | Fast pre-filter at detection time | Candidate evaluation before PSF fitting |

Note that moment-based morphological measures like the FWHM are biased when
image cutouts contain many background-dominated pixels. For this reason we
recommend the `half_width` keyword argument to `measure_star_shapes` should
not be too large; $\mathrm{half\_width} = \mathrm{FWHM}$ is often a reasonable value.
The returned `fwhm.y` and `fwhm.x` are row/column marginal widths, not
principal-axis widths; for a rotated source, use `fwhm.theta` only as the
covariance major-axis orientation.

```@docs
measure_star_shape
measure_star_shapes
```

### Moment Normalization and Aperture Sums

The aperture result includes `moment_norm`, the weighted zeroth moment
``M_{00}`` used internally to normalize centroids, FWHM estimates, and
aperture roundness.  This value follows the same weighting and masking as
the shape moments.  If `inv_var` is non-uniform, `moment_norm` is not a
physical source flux and should not be used for magnitude calibration or
as a photometric prior.

The result also includes `aperture_sum`, `aperture_area`, and
`aperture_sum_err`.  These are quick rectangular-cutout diagnostics over
unmasked pixels (`inv_var > 0`): `aperture_sum` is the unweighted sum of
`image - background`, `aperture_area` is the number of contributing
pixels, and `aperture_sum_err` is the formal propagated uncertainty
``\sqrt{\sum 1/\mathtt{inv\_var}}`` under the assumption of independent
pixel errors.  Mask invalid pixels by setting their inverse variance to
zero.  They are useful flux proxies, but are not aperture-corrected and
should not be used for anything requiring precision.
We generally prefer the `matched_filter_flux`, an explicit aperture
photometry routine, or a PSF-fit flux for calibrated photometry.

## References
This page cites the following references:

```@bibliography
Pages = ["morphology.md"]
Canonical = false
```
