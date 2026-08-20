```@meta
CurrentModule = CrowdPhot.PSF
```

# PSF API

## Abstract Interface

All PSF models in CrowdPhot are subtypes of `AbstractPSFModel{T}`. The
abstract interface defines the operations that every model must (or should)
support.

```@docs
AbstractPSFModel
```

### Core Evaluation

```@docs
evaluate
evaluate_fg
```

!!! note
    Note that all PSF models should have `evaluate` methods that are
    SIMD vectorizable via [LoopVectorization.@turbo](https://github.com/JuliaSIMD/LoopVectorization.jl).
    The functions [`add_star!`](@ref), [`subtract_star!`](@ref), and [`render`](@ref) internally use
    `evaluate(model, y, x)` inside loops declared vectorizable with `@turbo` and these will fail
    if you define a PSF model with an `evaluate` method that is not safely vectorizable. The most
    common cases of such failures are PSF models that use special functions. For an example of how
    to support such PSFs, see [`AiryPSF`](@ref) which uses bessel functions. In this case,
    we took pure-Julia bessel function implementations from [Bessels.jl](https://github.com/JuliaMath/Bessels.jl)
    and adapted them to make them branchless, making them safe for SIMD vectorization through `@turbo`.

### Model Properties

```@docs
centroid
integral
background
peak
amplitude
effective_area
fwhm
theta
```

### Spatial Utilities

```@docs
extent
ellipse_bounds
render
add_star!
subtract_star!
```

## API Internals

```@docs
free_params
model_from_vector
```
