using SafeTestsets
using Test

# Documenter evaluates @meta blocks in documentation source files
# (CurrentModule, etc.) via Core.eval(Main, ...).
# When run under @safetestset, each test file gets its own anonymous module, so
# imports in doctests.jl don't create bindings in Main.  We import here so that
# CrowdPhot is visible where Documenter expects it.
import CrowdPhot

# Run the PSF API, parametric fitting, and empirical PSF coverage together.
@safetestset "Doctests" include("doctests.jl")
@safetestset "PSF API tests" include("psf/common_psf_tests.jl")
@safetestset "PSF Fitting" include("psf/psf_fitting_tests.jl")
@safetestset "PSF Fit Parity" include("psf/psf_fit_parity_tests.jl")
@safetestset "Empirical PSF models" include("psf/empirical_model_tests.jl")
@safetestset "Simulation tests" include("simulation_test.jl")
@safetestset "Background estimation" include("background_tests.jl")
@safetestset "Centroids" include("centroids_tests.jl")
@safetestset "Correlation" include("correlation_tests.jl")
@safetestset "Detection" include("detection_tests.jl")
