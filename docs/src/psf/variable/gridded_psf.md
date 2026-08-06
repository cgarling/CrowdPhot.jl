```@meta
CurrentModule = CrowdPhot.PSF
```

# [GriddedPSFModel — spatially-varying PSF via grid interpolation](@id gridded_psf)

## Overview

Real optical systems do not produce the same PSF everywhere across a
detector: the shape typically varies with field position due to optical
aberrations, focus variation, and other effects. [`GriddedPSFModel`](@ref)
represents this spatial variation by tabulating a "node" PSF model at a
handful of fiducial detector positions and interpolating between them at
arbitrary evaluation points.

This mirrors the design of
[`photutils.psf.GriddedPSFModel`](https://photutils.readthedocs.io/en/stable/api/photutils.psf.GriddedPSFModel.html),
which is used by tools such as [`romanisim`](https://github.com/spacetelescope/romanisim)
to model spatially-varying effective PSFs (ePSFs) built from CRDS reference
files for the Roman Space Telescope. `GriddedPSFModel` in `CrowdPhot.jl`
follows the same bilinear-interpolation algorithm, but is generic with
respect to the *type* of PSF model tabulated at each node: node PSFs can be
[`ImagePSF`](@ref) (as in the `photutils`/`romanisim` use case), any
analytic model such as [`GaussianPRF`](@ref), or a mix of
concrete node types sharing the same numeric element type (though this may incur
a performance penalty due to the mixed container type).

## The interpolation grid

A `GriddedPSFModel` is built from:

- `psfs`: a vector of node PSF models, each pre-centered and normalized to
  unit flux/zero background at its own fiducial position,
- `grid_y`, `grid_x`: the detector `(y, x)` position of each node.

The node positions must form a **complete rectangular grid**: every
combination of the unique `grid_y` values and unique `grid_x` values must be
present exactly once. This mirrors the `photutils` requirement and allows
the evaluator to always locate the (up to four) nodes bounding any query
position via a simple sorted-array search, rather than a general 2D
triangulation. Node spacing need not be uniform. As a special case, a single
node (`length(psfs) == 1`) is always accepted and used everywhere (no
interpolation is performed).

`GriddedPSFModel` itself only has four fittable fields: `y`, `x` (the
detector position at which the model is currently centered/evaluated),
`flux`, and `bkg`, just like the other `AbstractPSFModel` types. Unlike
`photutils.psf.GriddedPSFModel`, there are no `oversampling`, `origin`, or
`fill_value` fields at the grid level: these are properties of how each
*node* PSF is tabulated (for example, an [`ImagePSF`](@ref) node already
carries its own `oversampling`, `origin`, and `fill_value`), so they are
naturally handled per-node instead of being duplicated on the container.

!!! note "Homogeneous element type"
    All node PSFs must currently share the same element type `T` (e.g. all
    `Float64`). Mixing, say, `Float32` and `Float64` nodes in the same grid
    is rejected at construction time. A future version may support
    automatic promotion of heterogeneous node types to a common `T`.

## Evaluation: bilinear blending

When `evaluate(model, y, x)` is called, the model:

1. Locates the grid cell containing `(model.y, model.x)` -- the up to four
   node PSFs at the corners of that cell (or, outside the grid bounds, the
   nearest edge cell).
2. Recenters each active corner node to `(model.y, model.x)` with unit flux
   and zero background.
3. Evaluates each recentered node at the query position `(y, x)` and
   combines the results with bilinear weights based on how close
   `(model.y, model.x)` is to each corner.
4. Scales the blended, unit-flux result by `model.flux` and adds
   `model.bkg`.

Because only the recentered *shape* interpolation depends on
`(model.y, model.x)`'s position relative to the grid, evaluating the model
at some other pixel `(y, x)` (as opposed to the model's own centroid)
proceeds exactly like evaluating any other `AbstractPSFModel`: the model
answers "what is the PSF value at pixel `(y, x)`, given that the star is
centered at `(model.y, model.x)`?".

Positions outside the overall node grid are handled by clamping into the
nearest edge cell for the purpose of computing interpolation weights. This
freezes the interpolated PSF *shape* at the grid boundary, while the model
can still be evaluated (and fit) at any true `(y, x)`.

`evaluate_fg` provides the exact analytic gradient of this bilinear blend
with respect to `y`, `x`, `flux`, and `bkg` (including the weight-derivative
terms from the interpolation itself, not just each node's own gradient),
so `GriddedPSFModel` is fully compatible with [`fit_star`](@ref) and the
rest of the generic `AbstractPSFModel` fitting/rendering machinery
(`extent`, `render`, `add_star!`, `subtract_star!`, `centroid`, `integral`,
`background`, ...) with no additional integration work required.

## Node model requirements

Any node PSF type can be used, as long as it:

- Supports `ConstructionBase.setproperties` with `y`, `x`, `flux`, and `bkg`
  keys (true of every `AbstractPSFModel` in this package), and
- Orders its `ConstructionBase.getproperties` (and, if fitting is needed,
  its `evaluate_fg` gradient) as `y, x, ..., flux, bkg` -- the convention
  followed by every model in `CrowdPhot.jl`.

This is validated once, at `GriddedPSFModel` construction time, so
malformed node types are rejected immediately with an informative error
rather than failing obscurely during evaluation or fitting.

## Example

```jldoctest
julia> using CrowdPhot.PSF: GriddedPSFModel, CircularGaussianPRF, evaluate

julia> nodes = [CircularGaussianPRF(y=y0, x=x0, fwhm=fwhm, flux=1.0, bkg=0.0)
                for ((y0, x0), fwhm) in zip(((0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0)), (2.0, 4.0, 4.0, 6.0))];

julia> model = GriddedPSFModel(nodes, [0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 0.0, 10.0]; y=5.0, x=5.0, flux=100.0, bkg=1.0);

julia> evaluate(model, 5.0, 5.0) > 1.0 # non-trivial blended PSF value above background
true
```

## Public API

```@docs
GriddedPSFModel
```

## Supporting Methods

The following methods are part of the `AbstractPSFModel` interface and work
identically for `GriddedPSFModel` as for other model types:

| Method | Description |
|---|---|
| `evaluate(model, y, x)` | Bilinear blend of the (up to four) active node PSFs |
| `evaluate_fg(model, y, x)` | Analytic gradient wrt `y`, `x`, `flux`, `bkg` |
| `centroid(model)` | Return `(model.y, model.x)` |
| `integral(model)` | Return `model.flux` |
| `background(model)` | Return `model.bkg` |
| `extent(model)` | Union of the active corners' own (recentered) `extent` |
