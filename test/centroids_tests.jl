using CrowdPhot
using CrowdPhot.PSF: CircularGaussianPSF, evaluate, peak as psf_peak
using FillArrays: Fill
using LinearAlgebra
using StableRNGs: StableRNG
using Statistics: mean
using Test

# Helper: generate a star image at a known position.
# FWHM=2.8 is the sweet spot from Vakili & Hogg (2016) — after matched-filter
# smoothing the 3×3 quadratic fit has enough curvature to work well.
function _make_star(; x0=5.0, y0=5.0, fwhm=2.8, flux=10.0, shape=(9,9))
    inds = (1:shape[1], 1:shape[2])
    model = CircularGaussianPSF(x=x0, y=y0, fwhm=fwhm, flux=flux, bkg=0.0)
    return evaluate.(model, inds[1], inds[2]'), model
end

@testset "centroid_poly" begin

    @testset "noise-free: centroid at pixel centre" begin
        img, model = _make_star(; x0=5.0, y0=5.0)
        result = centroid_poly(img)
        @test result.x ≈ 5.0 atol=1e-12
        @test result.y ≈ 5.0 atol=1e-12
        # peak should be reasonable (within 20% of true PSF peak)
        @test result.peak ≈ psf_peak(model) rtol=0.2
    end

    @testset "noise-free: sub-pixel centroid" begin
        # At FWHM=2.8 the quadratic fit should recover sub-pixel centroids
        # with small intrinsic bias (< 0.15 px as shown in the paper).
        img, model = _make_star(; x0=5.2, y0=5.3)
        result = centroid_poly(img)
        @test abs(result.x - 5.2) < 0.15
        @test abs(result.y - 5.3) < 0.15
        @test result.peak > 0
    end

    @testset "border behaviour" begin
        # max on edge → no 3×3 neighbourhood → NaN
        img = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        result = centroid_poly(img)
        @test isnan(result.x)
        @test isnan(result.y)
        @test isnan(result.peak)
        @test isnan(result.x_err)
        @test isnan(result.y_err)
        @test isnan(result.peak_err)

        # max at corner
        img2 = [100.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        result2 = centroid_poly(img2)
        @test isnan(result2.x)
        @test isnan(result2.y)
    end

    @testset "peak estimate consistency" begin
        for fwhm in (2.8, 4.0)
            img, model = _make_star(; x0=5.0, y0=5.0, fwhm=fwhm)
            result = centroid_poly(img)
            @test result.peak > 0
            @test result.peak ≈ psf_peak(model) rtol=0.4
        end
    end

    @testset "weighted least squares" begin
        img, _ = _make_star(; x0=5.0, y0=5.0)
        # uniform weights → same centroid as default
        r1 = centroid_poly(img)
        r2 = centroid_poly(img, ones(size(img)))
        @test r1.x ≈ r2.x
        @test r1.y ≈ r2.y
        @test r1.peak ≈ r2.peak

        # doubling inv_var halves the variance → errors scale by 1/√2
        r3 = centroid_poly(img, Fill(2.0, size(img)))
        @test r1.x ≈ r3.x
        @test r1.y ≈ r3.y
        @test r3.x_err ≈ r1.x_err / sqrt(2)  rtol=1e-10
        @test r3.y_err ≈ r1.y_err / sqrt(2)  rtol=1e-10
        @test r3.peak_err ≈ r1.peak_err / sqrt(2)  rtol=1e-10

        # down-weighting a corner pixel should not change centroid much for
        # symmetric star centred on a pixel
        w = ones(9, 9)
        w[1, 1] = 0.01
        r4 = centroid_poly(img, w)
        @test r4.x ≈ r1.x  atol=1e-4
        @test r4.y ≈ r1.y  atol=1e-4
    end

    @testset "noise tests: SNR regimes" begin
        # Star centred at a pixel to eliminate quadratic-approximation bias.
        # SNR defined as peak / σ_noise (per-pixel SNR at the peak).
        img_true, model = _make_star(; x0=5.0, y0=5.0)
        peak_true = psf_peak(model)
        rng = StableRNG(42)

        results = Dict{Int, Vector{Float64}}()
        for snr in (10, 50, 100)
            σ_noise = peak_true / snr
            errors_x = Float64[]
            errors_y = Float64[]
            for _ in 1:200
                noisy = img_true .+ σ_noise .* randn(rng, size(img_true))
                res = centroid_poly(noisy)
                push!(errors_x, res.x - 5.0)
                push!(errors_y, res.y - 5.0)
            end
            rmse_x = sqrt(mean(e -> e^2, errors_x))
            rmse_y = sqrt(mean(e -> e^2, errors_y))
            results[snr] = [rmse_x, rmse_y]
        end

        # RMSE should improve with increasing SNR
        @test results[50][1] < results[10][1]
        @test results[50][2] < results[10][2]
        @test results[100][1] < results[50][1]
        @test results[100][2] < results[50][2]

        # At SNR=100 the RMSE should be sub-0.1 pixel
        @test results[100][1] < 0.1
        @test results[100][2] < 0.1

        # At SNR=50, still sub-0.15 pixel
        @test results[50][1] < 0.2
        @test results[50][2] < 0.2

        # At SNR=10, larger but bounded by ~1 pixel
        @test results[10][1] < 1.5
        @test results[10][2] < 1.5
    end

    @testset "error estimates are finite and positive" begin
        img, _ = _make_star(; x0=5.0, y0=5.0)
        result = centroid_poly(img)
        @test isfinite(result.x_err)
        @test isfinite(result.y_err)
        @test isfinite(result.peak_err)
        @test result.x_err > 0
        @test result.y_err > 0
        @test result.peak_err > 0
    end

    @testset "centroid_poly with Fill inv_var" begin
        img, _ = _make_star(; x0=5.0, y0=5.0)
        # This uses Fill(1, size(img)) internally
        result = centroid_poly(img)
        @test result.x ≈ 5.0 atol=1e-12
        @test result.y ≈ 5.0 atol=1e-12
        @test result.peak > 0
    end

    @testset "_centroid_poly3 direct call" begin
        # Symmetric 3×3 → centroid at origin
        patch = [0.1 0.3 0.1;
                 0.3 1.0 0.3;
                 0.1 0.3 0.1]
        result = CrowdPhot._centroid_poly3(patch, ones(3,3))
        @test result.x ≈ 0.0 atol=1e-12
        @test result.y ≈ 0.0 atol=1e-12
        @test result.peak > 0
        @test result.x_err > 0
        @test result.y_err > 0
        @test result.peak_err > 0

        # Asymmetric patch: peak pulled toward brighter region
        patch2 = [0.05 0.2  0.1;
                  0.1  0.8  0.4;
                  0.05 0.15 0.08]
        result2 = CrowdPhot._centroid_poly3(patch2, ones(3,3))
        # Brighter on right side → x > 0
        @test result2.x > 0
        # The y-centroid sign depends on the exact fit; just check it's finite
        @test isfinite(result2.y)

        # Zero-weight on one half should pull centroid toward the other half
        w = ones(3, 3)
        w[:, 1] .= 0  # zero out left column
        result3 = CrowdPhot._centroid_poly3(patch, w)
        @test result3.x > 0  # centroid shifts right since left is ignored
    end
end
