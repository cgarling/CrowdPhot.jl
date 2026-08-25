```@meta
CurrentModule = CrowdPhot
```

# Detection and Centroiding

## Matched Filter Detection

CrowdPhot implements the formally correct matched filter for point-source
detection under stationary (uncorrelated) Gaussian noise.

### Mathematical Derivation

Consider an image ``D(y,x)`` (indexed as ``D[\mathrm{row}, \mathrm{col}]``)
consisting of a background ``B`` (either already subtracted, i.e. ``B = 0``,
or an unknown constant we will cancel below), a point source with flux ``F``
and PSF shape ``P(y,x)`` (normalized such that ``\sum P = 1``), and
independent Gaussian noise with a **position-dependent** variance:

```math
D(y,x) = B + F \cdot P(y - y_0, x - x_0) + N(y,x), \qquad
N(y,x) \sim \mathcal{N}\!\left(0,\, \sigma^2(y,x)\right).
```

Spatially varying noise is the practically relevant case -- sky
background, exposure depth, and detector gain/read-noise all vary across
real images -- so the derivation below is carried out for a general
inverse-variance map ``w(y,x) = 1/\sigma^2(y,x)`` from the start. The
spatially uniform case (``w`` constant) is recovered afterward as a
special case.

At each position we test the hypothesis ``\mathcal{H}_1`` (source present
with flux ``F``) versus ``\mathcal{H}_0`` (no source). For independent,
non-identically-distributed Gaussian noise the log-likelihood ratio is

```math
\ln\Lambda = F \sum_i w_i P_i (D_i - B)
           - \frac{F^2}{2} \sum_i w_i P_i^2 .
```

Maximizing with respect to ``F`` gives the inverse-variance-weighted
matched-filter flux estimator:

```math
\hat{F} = \frac{\sum_i w_i P_i (D_i - B)}{\sum_i w_i P_i^2} .
```

This expression requires both the background ``B`` and the weight map
``w_i`` to be known.

#### Canceling an unknown constant background: the zero-sum kernel

In practice the image may not be background-subtracted. Rather than
estimating ``B`` explicitly, we can fold the cancellation of a
**constant** background into the kernel itself by enforcing
``\sum K_i = 0``. This removes any offset that is **uniform** across the
kernel footprint. It does *not* remove gradients, curvature, or other
small-scale background structure; a proper background subtraction
should still be performed before detection. Define the **zero-sum
kernel**

```math
K_i = \frac{P_i - \bar{P}}{\mathrm{denom}}, \qquad
\bar{P} = \frac{1}{N}\sum_j P_j, \qquad
\mathrm{denom} = \sum_j P_j^2 - \frac{(\sum_j P_j)^2}{N} .
```

This kernel satisfies ``\sum K_i = 0`` and ``\sum K_i P_i = 1``, so
``E\!\left[\sum K_i D_i \mid \mathcal{H}_1\right] = F`` for an isolated
source: this cancellation is a statement about the expectation of an
**unweighted** correlation and does not involve ``w_i`` at all, so the
zero-sum flux response is unbiased regardless of how the noise variance
is distributed across the footprint. `CrowdPhot` always builds ``K``
from the unweighted template ``P`` -- it is never locally reweighted by
`inv_var` (see the note below for why, and for the caveat this implies).

#### The detection statistic

For a general (possibly spatially varying) inverse-variance map ``w_i``,
define two correlations of the zero-sum kernel with the image:

```math
\mathrm{num}(y,x) = \sum_{i,j} K_{i,j} \, w_{y+i,\,x+j} \, D_{y+i,\,x+j},
\qquad
\mathrm{den}(y,x) = \sum_{i,j} K_{i,j}^2 \, w_{y+i,\,x+j} .
```

The **detection statistic** is

```math
\mathrm{SNR}(y,x) = \frac{\mathrm{num}(y,x)}{\sqrt{\mathrm{den}(y,x)}} .
```

This is exactly what [`matched_filter`](@ref)'s weighted path (`inv_var`
supplied) computes: `num` and `den` are correlations of ``w \cdot D`` and
``w`` with ``K`` and ``K^2`` respectively.

**Why this statistic is meaningful.** Under ``\mathcal{H}_0``,
``D_i = B + N_i``. If the background term cancels -- either because the
image is already background-subtracted (``B = 0``), or because ``w`` is
constant over the kernel footprint (so ``\sum_i K_i w_i B = Bw\sum_i K_i
= 0``) -- then

```math
\mathrm{num}(y,x) \Big|_{\mathcal{H}_0} = \sum_i K_i w_i N_i .
```

Since the ``N_i`` are independent with ``\operatorname{Var}[N_i] =
1/w_i``,

```math
\operatorname{Var}\!\left[\mathrm{num} \mid \mathcal{H}_0\right]
 = \sum_i K_i^2 w_i^2 \cdot \frac{1}{w_i}
 = \sum_i K_i^2 w_i
 = \mathrm{den}(y,x),
```

so

```math
\mathrm{SNR} \mid \mathcal{H}_0 \sim \mathcal{N}(0, 1) .
```

!!! warning "This requires the background to actually cancel"
    When the image is already background-subtracted (``B = 0``), the
    result above holds for **any** ``K`` and **any** weight map ``w`` --
    the background term is identically zero, so ``\sum K_i = 0`` isn't
    even needed for calibration (it still matters for the flux estimate,
    see above). But if ``B \neq 0`` (not background-subtracted) **and**
    ``w`` varies across the kernel footprint, ``\sum_i K_i w_i B \neq 0``
    in general -- the unweighted zero-sum kernel only guarantees
    ``\sum K_i = 0``, not ``\sum K_i w_i = 0``. In that case `SNR` is
    **not** calibrated to ``\mathcal{N}(0,1)`` under the null; see the
    note on kernel reweighting below and the recommended-configurations
    table.

Under ``\mathcal{H}_1`` (source of flux ``F`` present, background
canceled),

```math
E[\mathrm{num} \mid \mathcal{H}_1] = F \sum_i K_i w_i P_i, \qquad
E[\mathrm{SNR} \mid \mathcal{H}_1] =
\frac{F \sum_i K_i w_i P_i}{\sqrt{\sum_i K_i^2 w_i}} .
```

Sources bright enough to push this expectation well above the chosen
threshold are reliably detected; fainter sources fall below it.

#### Special case: spatially uniform noise

When the noise is spatially uniform, ``w_i \equiv 1/\sigma^2`` is
constant over the kernel footprint. Substituting into the definitions
above,

```math
\mathrm{num}(y,x) = w \sum_i K_i D_i = w\,S(y,x), \qquad
\mathrm{den}(y,x) = w \sum_i K_i^2 = \frac{w}{\mathrm{denom}},
```

where ``S = K \star D`` is the (unweighted) correlation of the image
with the kernel, and the second equality uses ``\sum_i K_i^2 = \sum_i
(P_i - \bar P)^2 / \mathrm{denom}^2 = 1/\mathrm{denom}``. The general
`SNR(y,x)` therefore collapses to

```math
\mathrm{SNR}(y,x)\Big|_{w \,=\, 1/\sigma^2}
       = \frac{w\,S(y,x)}{\sqrt{w/\mathrm{denom}}}
       = \frac{S(y,x)\,\sqrt{\mathrm{denom}}}{\sigma} .
```

This is `SNR(y,x)`
evaluated for a constant weight, algebraically simplified so that
``\sigma`` appears explicitly rather than being folded into ``w``. All
properties derived above for `SNR` carry over directly: with ``w``
constant, ``\sum_i K_i w_i = w \sum_i K_i = 0`` automatically, so the
background always cancels and

```math
SNR \mid \mathcal{H}_0 \sim \mathcal{N}(0, 1), \qquad
E[SNR \mid \mathcal{H}_1] = \frac{F \sqrt{\mathrm{denom}}}{\sigma}
```

(the ``\mathcal{H}_1`` expectation follows from ``\sum_i K_i P_i = 1``
above, since ``w`` factors out of ``\sum_i K_i w P_i / \sqrt{w/\mathrm{denom}}``).
A threshold of `SNR = 5` therefore corresponds to a false-positive
probability of ``\sim 3\times10^{-7}`` per independent resolution
element (one-sided Gaussian tail). Sources with ``F \gg \sigma /
\sqrt{\mathrm{denom}}`` produce large `SNR` values and are detected;
fainter sources fall below the threshold.

In the uniform-weight code path (`inv_var = nothing`), [`matched_filter`](@ref)
does not take a scalar noise estimate, so it does not estimate `SNR` directly.
It stores and thresholds ``S\sqrt{\mathrm{denom}}``, which equals
``\sigma SNR``. Divide that map by your pixel-noise estimate, or provide
`inv_var = fill(1/sigma^2, size(image))` (which routes through the
general weighted path above and returns `SNR` -- numerically identical -- directly),
to obtain a map in standard-deviation units.

!!! note "Kernel Reweighting for Spatially-Variable Weights"
    If an unknown constant background is also being profiled out while
    the weights vary across the footprint, the background projection
    should be weighted locally:

    ```math
    \bar{P}_w(y,x) =
    \frac{\sum_{i,j} w_{y+i,\,x+j} P_{i,j}}
        {\sum_{i,j} w_{y+i,\,x+j}},
    \qquad
    \mathrm{denom}_w(y,x) =
    \sum_{i,j} w_{y+i,\,x+j}
    \left(P_{i,j} - \bar{P}_w(y,x)\right)^2.
    ```

    The corresponding profiled-background statistic is

    ```math
    \mathrm{SNR}_w(y,x) =
    \frac{\sum_{i,j} w_{y+i,\,x+j}
        \left(P_{i,j} - \bar{P}_w(y,x)\right)
        D_{y+i,\,x+j}}
        {\sqrt{\mathrm{denom}_w(y,x)}} .
    ```

    When the sky background dominates the noise and is approximately
    constant, ``w`` is approximately constant over the kernel footprint,
    so this reduces to the zero-sum kernel above. If ``w`` varies
    strongly, ``\sum K_i = 0`` alone does not cancel a constant
    background inside ``\sum K_i w_i D_i`` (per the warning above). For
    simplicity and speed we **do not** apply this kernel reweighting,
    which means that if you have an image with significant spatial
    variation in the noise properties, you should **always** model and
    subtract the background before running detection.

The zero-sum kernel is the default and the recommended choice: even on
a background-subtracted image, ``\sum K_i = 0`` cancels any residual
uniform offset. For an isolated source matched by ``P``, the expected flux
response remains unbiased, although the form of the estimator changes slightly
and the variance is slightly larger. The
`normalize_zerosum` parameter can be set to `false` to skip the
zero-sum correction, which yields marginally lower noise variance at
the cost of sensitivity to imperfect background subtraction.

The zero-sum constraint does carry a small noise penalty, exact for
spatially uniform noise (and approximate when ``w`` varies but is
roughly constant over the kernel footprint -- the exact penalty for
strongly varying ``w`` depends on the correlation between ``w`` and
``P`` and is not captured by the closed form below). For uniform
noise, the variance of the correlation response is
``\sigma^2/\mathrm{denom}`` rather than the ``\sigma^2/\sum P^2`` that
would be achieved without the zero-sum correction. The SNR ratio is

```math
\frac{\mathrm{SNR}_\mathrm{zs}}{\mathrm{SNR}_\mathrm{nzs}}
 = \sqrt{1 - \frac{(\sum P)^2}{N\sum P^2}} .
```

The penalty depends on the PSF width relative to the kernel truncation
radius, not on the pixel count alone. For a Gaussian PSF with width
``\sigma`` truncated at radius ``R``, the ratio is approximately
``\sqrt{1 - 4\sigma^2/R^2}``. At ``R = 4\sigma`` (typical) the
penalty is ~13%; at ``R = 3\sigma`` it is ~25%. Only for very
generously padded kernels (``R \gtrsim 10\sigma``) does it drop
below a few percent.

### Recommended configurations

| Image state | `inv_var` | `normalize_zerosum` | Rationale |
|---|---|---|---|
| Not background-subtracted, noise level unknown | `nothing` | `true` (default) | Zero-sum kernel cancels a uniform background automatically.  The significance map is in units of the (unknown) pixel-noise σ (RMS).  To renormalize it after calculation, divide by your σ estimate to obtain true SNR in standard deviations. |
| Not background-subtracted, uniform noise σ known | `fill(1/σ², size(image))` | `true` (default) | Zero-sum kernel cancels a uniform background automatically.  Significance map is in standard-deviation units (∼N(0,1) under the null). |
| Not background-subtracted, varying noise | -- | -- | Background-subtract the image first.  With an un-subtracted background and varying `inv_var`, neither kernel normalization is correct: `true` leaks background through ΣKᵢwᵢ ≠ 0, and `false` leaks through ΣKᵢ ≠ 0.  After subtraction, use the "Background-subtracted, varying noise" row below. |
| Background-subtracted, uniform noise σ known | `fill(1/σ², size(image))` | `false` | No residual background remains, so the zero-sum correction carries a variance penalty with no benefit.  Significance map is in standard-deviation units (∼N(0,1) under the null).  `true` also works but is noisier. |
| Background-subtracted, varying noise | `Matrix` | `false` | The known-background estimator (K = P/ΣP²) is correct for any weight map.  `true` also works but carries the zero-sum variance penalty with no benefit.  Significance map is in standard-deviation units (∼N(0,1) under the null). |

Background and RMS error estimates can be measured by tools described in the [`Background`](@ref bkg)
section.

### The `rel_err` term

The DAOPHOT family of codes [Stetson1987](@cite) -- and
[photutils'](https://github.com/astropy/photutils)
`DAOStarFinder`, which follows the same formalism -- define

```math
\mathrm{rel\_err} = \frac{1}{\sqrt{\mathrm{denom}}},
\qquad
\mathrm{denom} = \sum_i P_i^2 - \frac{(\sum_i P_i)^2}{N}.
```

This quantity is formally correct for matched-filter detection.
It arises directly from the variance of the convolved image under
``\mathcal{H}_0``.  With the zero-sum kernel
``K_i = (P_i - \bar{P})/\mathrm{denom}``,

```math
\operatorname{Var}[K \star D \mid \mathcal{H}_0]
   = \sigma^2 \sum_i K_i^2
   = \frac{\sigma^2}{\mathrm{denom}}
   = (\sigma \cdot \mathrm{rel\_err})^2.
```

The photutils detection threshold is therefore
``\mathtt{threshold} = n_\sigma \cdot \sigma \cdot \mathrm{rel\_err}``,
which equals ``n_\sigma`` times the standard deviation of the convolved
noise.

CrowdPhot does **not** expose `rel_err` as a separate quantity.
Instead, the ``\mathrm{rel\_err}`` factor is folded into the
`significance_map` computation.  In the uniform-weight path
(`inv_var = nothing`), the map is formed as

```math
\mathtt{significance\_map} = (K \star D) \cdot \sqrt{\mathrm{denom}}
                            = \frac{K \star D}{\mathrm{rel\_err}},
```

which equals the matched-filter SNR up to the unknown factor of
``1/\sigma`` (divide by your noise estimate to obtain the true SNR).
In the weighted path (`inv_var` supplied), the division by
``\sqrt{\mathrm{den}}`` in the expression
``\mathrm{SNR} = \mathrm{num} / \sqrt{\mathrm{den}}`` plays the
same role, with the noise variance coming from the data rather than
from a single scalar ``\sigma``.  In both cases the `rel_err` is
present -- it simply appears as ``1/\sqrt{\mathrm{denom}}`` inside
the kernel normalization rather than as a separately stored
parameter.

### API

[`matched_filter`](@ref) is the entry point for matched filter source
detection. The input for the `kernel` argument can be
- a generic `AbstractMatrix` following the conventions expected by [`CrowdPhot.correlate`](@ref);
- an instance of an `AbstractPSFModel` such as a [`MoffatPSF`](@ref);
  the [`render`](@ref) function will be called on the model and the 
  resulting matrix will be used as the kernel.
- an `Int` giving the FWHM of a [`CircularGaussianPRF`](@ref) model to use for the
  correlation;
- a `Tuple{Int, Int}` giving the ``(y_\mathrm{FWHM}, x_\mathrm{FWHM})`` of a
  general [`GaussianPRF`](@ref) model to use for the correlation.

```@docs
matched_filter
MatchedFilterResult
CrowdPhot.findlocalmaxima
CrowdPhot.correlate
CrowdPhot.correlate!
```

## References
This page cites the following references:

```@bibliography
Pages = ["detection.md"]
Canonical = false
```
