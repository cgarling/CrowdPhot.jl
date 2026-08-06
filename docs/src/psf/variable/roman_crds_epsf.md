```@meta
CurrentModule = CrowdPhot.PSF
```

# [Reading Roman CRDS ePSF reference files](@id roman_crds_epsf)

## Overview

[`roman_crds_gridded_epsf`](@ref) reads a Roman Space Telescope CRDS ePSF
reference file (an `.asdf` file, as delivered by CRDS, e.g.
`roman_wfi_epsf_*.asdf`) into a [`GriddedPSFModel`](@ref) of [`ImagePSF`](@ref)
nodes, following the same tabulated-node/bilinear-interpolation design used by
`romanisim` and `romancal` for the same reference files.

This function requires [`ASDF.jl`](https://github.com/JuliaAstro/ASDF.jl) to
be loaded:

```julia
using CrowdPhot.PSF: roman_crds_gridded_epsf
using ASDF  # required: activates the CrowdPhotASDFExt package extension

model = roman_crds_gridded_epsf("roman_wfi_epsf_0182.asdf")
```

`ASDF.jl` is a **weak dependency** of `CrowdPhot.jl`: it is not installed or
loaded unless you explicitly `using ASDF` yourself. This keeps `ASDF.jl` (and
its own dependency tree) out of the load path for users who never need to
read Roman CRDS reference files, while still making the reader available
as soon as both packages are loaded together in the same session, via
Julia's package extension mechanism.

## Selecting a node PSF slice

A CRDS ePSF reference file tabulates PSFs on a grid of detector positions,
for several **spectral types**, and for several degrees of **defocus** -- a
5-dimensional array. `roman_crds_gridded_epsf` selects one 2D stamp per grid
node (for a chosen `spectral_type` and `defocus`) to build the
`GriddedPSFModel`:

```julia
model = roman_crds_gridded_epsf(path; spectral_type = "G2V", defocus = 0, psf_subtype = "psf")
```

- `spectral_type` (default `"G2V"`) selects the spectral-type slice by name,
  matching the file's own `meta.spectral_type` list (typically
  `["A0V", "G2V", "M5V"]`). The default matches
  [`romanisim`](https://github.com/spacetelescope/romanisim)'s own default
  PSF selection.
- `defocus` (default `0`, i.e. in-focus) selects the defocus-waves slice by
  value, matching the file's own `meta.defocus` list.
- `psf_subtype` (default `"psf"`) selects between `"psf"` (includes the
  detector's interpixel-capacitance response) and `"psf_noipc"` (without
  it). The single, non-gridded `"extended_psf"`/`"extended_psf_noipc"`
  stamps are not supported by `GriddedPSFModel` (no grid of positions);
  build an [`ImagePSF`](@ref) directly from `ASDF.load(path)["roman"]["extended_psf"][]`
  if you need one of those.

An invalid `spectral_type`, `defocus`, or `psf_subtype` raises an
`ArgumentError` listing the values actually available in the file.

## `oversample` is not a keyword argument

Every node `ImagePSF`'s `oversampling` is always read from the file's own
`meta.oversample` field, and is **not** an overridable keyword on
`roman_crds_gridded_epsf`. This is a fact about how the file's stamps were
tabulated (how many oversampled subpixels correspond to one native detector
pixel), not a user preference: passing any other value would silently
mismap the stamp's coordinate grid, distorting the effective PSF rather than
resampling it. If you need a genuinely different effective oversampling
(e.g. because your science image is binned relative to the reference
file's native pixel scale), that requires actually resampling the tabulated
array, which is out of scope for this reader -- construct the
`GriddedPSFModel` by hand instead (see below).

## Normalization and pixel-response convolution

CRDS ePSF reference files may store PSF stamps in one of two conventions:

- **Optical-PSF-only** ("old format"): the stamp is the raw optical PSF as
  computed by `stpsf`, *before* convolution with the detector's pixel
  response. Such stamps sum to approximately `1` (not `oversample^2`).
- **Pixel-integrated** ("new format"): the stamp already has the detector's
  pixel response convolved in, and sums to approximately `oversample^2`.

`roman_crds_gridded_epsf` detects which convention a file uses (following
the same heuristic as `romancal`'s `get_gridded_psf_model`: the median
per-node stamp sum, compared against `oversample^2 / 2`) and, for
old-format files only, convolves each node with the detector's pixel
response (a flat `oversample`-wide box, discretized the same way as
`astropy.convolution.Box2DKernel`) and rescales by `oversample^2` before
constructing the `ImagePSF`s. New-format stamps are used unchanged.

Each node `ImagePSF` is constructed with `normalize = false` (also
overridable via the `normalize` keyword): the array's own total, which is
typically slightly less than `oversample^2` (the small remaining deficit
being genuine flux in the wings beyond the finite stamp), is preserved
rather than forced to exactly `oversample^2`. This is believed to give
unbiased PSF-fitting fluxes on Roman's per-pixel-surface-brightness
(`PHOTMJSR`-style) calibration, without requiring a separate aperture
correction -- but this reasoning has not been independently confirmed
against official Roman documentation, and may be revisited.

## Escape hatch: building the model by hand

`roman_crds_gridded_epsf` intentionally only covers the common case. For
anything it does not support (a corrected `oversample`, the extended PSF,
a non-CRDS file layout, etc.), read the file directly and construct the
`GriddedPSFModel` yourself:

```julia
using ASDF
using CrowdPhot.PSF: ImagePSF, GriddedPSFModel

af = ASDF.load(path; extensions = true, validate_checksum = false)
afr = af["roman"]
data = afr["psf"][]  # Julia order: (x, y, grid_index, spectral_type, defocus)
stamps = [ImagePSF(permutedims(view(data, :, :, i, spectral_idx, defocus_idx));
                    oversampling = afr["meta"]["oversample"]) for i in 1:size(data, 3)]
model = GriddedPSFModel(stamps, afr["meta"]["pixel_y"], afr["meta"]["pixel_x"])
```

## Public API

```@docs
roman_crds_gridded_epsf
```
