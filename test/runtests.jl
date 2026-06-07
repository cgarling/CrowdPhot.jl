using SafeTestsets
using Test

# Run the PSF API, parametric fitting, and empirical PSF coverage together.
@safetestset "PSF API tests" include("psf/common_psf_tests.jl")
@safetestset "PSF Fitting" include("psf/psf_fitting_tests.jl")
@safetestset "Empirical PSF models" include("psf/empirical_model_tests.jl")
include("simulation_test.jl")
