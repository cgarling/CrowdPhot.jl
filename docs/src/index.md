```@meta
CurrentModule = CrowdPhot
```

# CrowdPhot.jl

CrowdPhot is a Julia package for crowded-field stellar photometry. It provides
analytic and empirical point-spread function (PSF) models, a Levenberg-Marquardt
optimizer with iteratively reweighted least squares (IRLS) for robust fitting,
and tools to simulate crowded stellar images.

## Package Overview

| Page | Description |
|------|-------------|
| [PSF API](psf/psf_api.md) | Abstract PSF interface and shared utilities |
| [Parametric PSF Models](psf/parametric_models.md) | Gaussian, Moffat, and Airy analytic models |
| [Effective PSF Models](psf/empirical/epsf_overview.md) | Empirical (image-backed) ePSF models |
| [Levenberg-Marquardt Fitter](lm_fitter.md) | LM optimizer with IRLS robust fitting |
| [Simulation](simulation.md) | Synthetic image and source generation |
