# using PrecompileTools: @setup_workload, @compile_workload

# @compile_workload begin
#     for T in (Float32, Float64)
#         # vv = [CircularGaussianPRF(y = T(gy), x = T(gx), fwhm = T(3.0), flux = T(1.0), bkg = T(0.0)) for (gy, gx) in ((0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0))]
#         # Note that different keyword arguments shapes (e.g., ommitting some) 
#         # require different compilations, so this precompiles
#         # only for the specific set of keyword arguments used here. If you want
#         # to precompile for other sets of keyword arguments, you need to call
#         # GriddedPSFModel with those arguments as well.
#         # psf = GriddedPSFModel(
#         #            vv,
#         #            T[0.0, 0.0, 10.0, 10.0], T[0.0, 10.0, 0.0, 10.0];
#         #             y = T(2.4), x = T(1.3), flux = T(120.0), bkg = T(10.0),
#         #         )

#         Background2D(randn(T, 50, 50), 5; coverage_mask=zeros(Bool, 50, 50), fill_value=T(NaN))
#         mf = matched_filter(zeros(T, 10, 10), zeros(T, 3, 3); inv_var=ones(T, 10, 10), sigma=T(5.0))
#         measure_star_shapes(mf; half_width=2)
#     end
# end

# Test for if Julia is precompiling:
# https://discourse.julialang.org/t/is-it-possible-to-detect-if-julia-is-ahead-of-time-precompiling/78631/3

# if ccall(:jl_generating_output, Cint, ()) == 1 

# Common calls *on* PSFs
for psf in (PSF.AiryPSF,)
    for T in (Float32, Float64)
        psfT = psf{T}
        precompile(evaluate, (psfT, T, T))
        precompile(render, (psfT,))
        precompile(PSF.add_star!, (Matrix{T}, psfT))
        precompile(PSF.subtract_star!, (Matrix{T}, psfT))
    end
end

for T in (Float32, Float64)
    precompile(AiryPSF, (T, T, T, T, T))
    precompile(CircularGaussianPSF, (T, T, T, T, T))
    precompile(CircularGaussianPRF, (T, T, T, T, T))
    precompile(GaussianPSF, (T, T, T, T, T, T, T))
    precompile(GaussianPRF, (T, T, T, T, T, T, T))
    precompile(CircularMoffatPSF, (T, T, T, T, T, T))
    precompile(MoffatPSF, (T, T, T, T, T, T, T, T))
    precompile(ImagePSF, (Matrix{T},))
    # precompile(GriddedPSFModel, (Vector{<:AbstractPSFModel}, Vector{T}, Vector{T}))
    # precompile(GriddedPSFModel, (Vector{CircularGaussianPRF{T}}, Vector{T}, Vector{T}))
    precompile(Core.kwcall, (NamedTuple{(:coverage_mask, :fill_value)}, typeof(Background2D), Matrix{T}, Int))
    precompile(Core.kwcall, (NamedTuple{(:inv_var, :sigma), Tuple{Matrix{T}, T}}, typeof(matched_filter), Matrix{T}, Matrix{T}))
    precompile(Core.kwcall, (NamedTuple{(:half_width,), Tuple{Int}}, typeof(measure_star_shapes), MatchedFilterResult{T}))
end

# end