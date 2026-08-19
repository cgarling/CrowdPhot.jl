# using PrecompileTools: @setup_workload, @compile_workload

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

# PSF construction
for T in (Float32, Float64)
    precompile(AiryPSF, (T, T, T, T, T))
end

# end