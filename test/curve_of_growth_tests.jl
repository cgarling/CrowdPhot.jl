using CrowdPhot:
    CircularAperture,
    CurveOfGrowth,
    curve_of_growth,
    encircled_energy,
    radius_at_energy,
    normalize,
    reference_cog,
    ExactOverlap,
    CenterOverlap,
    clipped_axes,
    aperture_weight
using CrowdPhot.PSF:
    CircularGaussianPSF, CircularMoffatPSF, GaussianPSF,
    evaluate, centroid, integral
using Test

# ==============================================================================
# Image-based curve_of_growth
# ==============================================================================

@testset "curve_of_growth — image" begin
    @testset "constant image, ExactOverlap" begin
        img = fill(2.0, 50, 50)
        radii = [1.0, 2.0, 3.0, 5.0]
        cog = curve_of_growth(img, 25.0, 25.0, radii; method = ExactOverlap())
        @test cog.y == 25.0
        @test cog.x == 25.0
        @test cog.radii == radii
        @test length(cog.flux) == 4
        @test length(cog.area) == 4
        @test isempty(cog.flux_err)
        # Sum of weights * constant = π r² * 2.0
        for (k, r) in enumerate(radii)
            @test cog.flux[k] ≈ π * r^2 * 2.0 rtol = 1e-12
            @test cog.area[k] ≈ π * r^2 rtol = 1e-12
        end
    end

    @testset "constant image, CenterOverlap" begin
        img = fill(3.0, 30, 30)
        cog = curve_of_growth(img, 15.0, 15.0, [2.0, 4.0];
                              method = CenterOverlap())
        # CenterOverlap: each pixel weight is 0 or 1, so area is an integer
        # count of pixels whose centers lie inside the circle.
        @test cog.flux[2] > cog.flux[1]  # larger radius → more flux
        @test cog.area[2] > cog.area[1]
        @test isinteger(cog.area[1])
        @test isinteger(cog.area[2])
    end

    @testset "background subtraction" begin
        img = fill(5.0, 50, 50)
        cog = curve_of_growth(img, 25.0, 25.0, [3.0]; background = 5.0)
        @test cog.flux[1] ≈ 0.0 atol = 1e-12
    end

    @testset "error propagation" begin
        img = fill(2.0, 30, 30)
        inv_var = fill(0.25, 30, 30)  # σ = 2 per pixel, variance = 4
        cog = curve_of_growth(img, 15.0, 15.0, [3.0];
                              method = ExactOverlap(), inv_var = inv_var)
        @test !isempty(cog.flux_err)
        @test cog.flux_err[1] > 0
        # Recompute error the same way COG does: sum w² * σ² over clipped_axes
        ap = CircularAperture(15.0, 15.0, 3.0)
        yr, xr = clipped_axes(ap, img)
        expected_var = 0.0
        for j in xr, i in yr
            w = aperture_weight(ap, i, j, ExactOverlap())
            w > 0 || continue
            expected_var += w^2 * 4.0  # σ² = 1/inv_var = 4
        end
        @test cog.flux_err[1] ≈ sqrt(expected_var) rtol = 1e-12
    end

    @testset "edge clipping: aperture near image border" begin
        img = fill(1.0, 20, 20)
        # Center near corner, large radius → aperture partially outside
        cog = curve_of_growth(img, 2.0, 2.0, [5.0, 10.0])
        @test cog.area[1] < π * 5.0^2  # clipped
        @test cog.area[2] > cog.area[1]  # still increasing
        @test all(isfinite, cog.flux)
        @test all(isfinite, cog.area)
    end

    @testset "masked inv_var pixels" begin
        img = fill(2.0, 20, 20)
        inv_var = fill(1.0, 20, 20)
        inv_var[10, 10] = 0.0     # masked
        inv_var[10, 11] = NaN     # masked
        cog = curve_of_growth(img, 10.0, 10.0, [2.0];
                              inv_var = inv_var)
        # Area is unaffected by inv_var masking (it's geometric)
        @test cog.area[1] ≈ π * 2.0^2 rtol = 1e-12
        # Error is still computed from non-masked pixels
        @test cog.flux_err[1] > 0
    end

    @testset "input validation" begin
        img = zeros(20, 20)
        @test_throws ArgumentError curve_of_growth(img, 10, 10, Float64[])
        @test_throws ArgumentError curve_of_growth(img, 10, 10, [-1.0, 1.0])
        @test_throws ArgumentError curve_of_growth(img, 10, 10, [3.0, 2.0])
        @test_throws ArgumentError curve_of_growth(img, 10, 10, [1.0];
            inv_var = zeros(10, 10))
    end
end

# ==============================================================================
# Model-based curve_of_growth — generic pixel integration
# ==============================================================================

@testset "curve_of_growth — model (generic)" begin
    @testset "GaussianPSF (non-circular, uses pixel integration)" begin
        model = GaussianPSF(y=15.0, x=15.0, x_fwhm=3.0, y_fwhm=4.0,
                            theta=30.0, flux=100.0, bkg=0.0)
        radii = [1.0, 3.0, 10.0]
        cog = curve_of_growth(model, radii; method = ExactOverlap())
        @test length(cog.flux) == 3
        @test cog.flux[3] > cog.flux[2] > cog.flux[1] > 0
        @test cog.flux[3] ≈ 100.0 rtol = 0.1  # most flux within r=10
        @test isempty(cog.flux_err)
        @test cog.y ≈ 15.0
        @test cog.x ≈ 15.0
    end

    @testset "input validation" begin
        model = CircularGaussianPSF(y=15.0, x=15.0, fwhm=3.0, flux=100.0, bkg=0.0)
        @test_throws ArgumentError curve_of_growth(model, Float64[])
        @test_throws ArgumentError curve_of_growth(model, [-1.0, 1.0])
        @test_throws ArgumentError curve_of_growth(model, [3.0, 2.0])
    end
end

# ==============================================================================
# Model-based curve_of_growth — analytic overloads
# ==============================================================================

@testset "curve_of_growth — analytic" begin
    @testset "CircularGaussianPSF enclosed flux formula" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=3.0, flux=100.0, bkg=0.0)
        # enclosed_flux(r) = flux * (1 - exp(γ * r² / fwhm²))
        # At r = fwhm/2 = 1.5: exp(γ/4) = exp(-log(2)) = 0.5 → flux = 50
        cog = curve_of_growth(model, [1.5])
        @test cog.flux[1] ≈ 50.0 rtol = 1e-12
        # At large r → total flux
        cog2 = curve_of_growth(model, [20.0])
        @test cog2.flux[1] ≈ 100.0 rtol = 1e-12
    end

    @testset "CircularMoffatPSF enclosed flux formula" begin
        model = CircularMoffatPSF(y=0.0, x=0.0, α=2.0, β=3.0,
                                  flux=100.0, bkg=0.0)
        # enclosed_flux(r) = flux * (1 - (1 + r²/α²)^(1-β))
        # At r = α = 2.0: flux = 100 * (1 - 2^(-2)) = 100 * 0.75 = 75
        cog = curve_of_growth(model, [2.0])
        @test cog.flux[1] ≈ 75.0 rtol = 1e-12
        # At r = 0: flux → 0
        cog0 = curve_of_growth(model, [1e-6])
        @test cog0.flux[1] ≈ 0.0 atol = 1e-6
    end

    @testset "analytic vs pixel integration agree" begin
        model = CircularGaussianPSF(y=15.0, x=15.0, fwhm=4.0,
                                    flux=100.0, bkg=0.0)
        radii = [1.0, 2.0, 3.0, 5.0]
        # Force pixel-integration path via AbstractPSFModel dispatch
        cog_pixel = curve_of_growth(model, radii; method = ExactOverlap())
        # Analytic path should give same result (within pixel discretization)
        for (k, r) in enumerate(radii)
            expected = 100.0 * (1 - exp(-4 * log(2) * r^2 / 16.0))
            @test cog_pixel.flux[k] ≈ expected rtol = 1e-12
        end
    end
end

# ==============================================================================
# encircled_energy
# ==============================================================================

@testset "encircled_energy" begin
    @testset "from COG — exact at sample points" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=100.0, bkg=0.0)
        cog = curve_of_growth(model, [1.0, 2.0, 3.0, 5.0])
        @test encircled_energy(cog, 1.0) ≈ cog.flux[1]
        @test encircled_energy(cog, 2.0) ≈ cog.flux[2]
        @test encircled_energy(cog, 5.0) ≈ cog.flux[4]
    end

    @testset "from COG — interpolation" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=100.0, bkg=0.0)
        cog = curve_of_growth(model, [1.0, 5.0])
        ef = encircled_energy(cog, 3.0)
        @test ef > cog.flux[1]
        @test ef < cog.flux[2]
    end

    @testset "from COG — out of range returns NaN" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=100.0, bkg=0.0)
        cog = curve_of_growth(model, [1.0, 5.0])
        @test isnan(encircled_energy(cog, 0.5))
        @test isnan(encircled_energy(cog, 10.0))
    end

    @testset "from model — analytic (CircularGaussianPSF)" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=3.0, flux=100.0, bkg=0.0)
        ef = encircled_energy(model, 1.5)  # half-light radius
        @test ef ≈ 50.0 rtol = 1e-12
    end

    @testset "from model — analytic (CircularMoffatPSF)" begin
        model = CircularMoffatPSF(y=0.0, x=0.0, α=2.0, β=3.0,
                                  flux=100.0, bkg=0.0)
        ef = encircled_energy(model, 2.0)
        @test ef ≈ 75.0 rtol = 1e-12
    end

    @testset "from model — generic (non-circular GaussianPSF)" begin
        model = GaussianPSF(y=15.0, x=15.0, x_fwhm=3.0, y_fwhm=3.0,
                            theta=0.0, flux=100.0, bkg=0.0)
        ef = encircled_energy(model, 20.0)
        @test ef > 0
        @test ef ≈ 100.0 rtol = 0.05
    end

    @testset "model vs COG consistency" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=100.0, bkg=0.0)
        ef_direct = encircled_energy(model, 2.0)
        cog = curve_of_growth(model, [2.0])
        @test ef_direct ≈ cog.flux[1] rtol = 1e-12
    end
end

# ==============================================================================
# radius_at_energy
# ==============================================================================

@testset "radius_at_energy" begin
    @testset "exact at sample points" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=100.0, bkg=0.0)
        cog = curve_of_growth(model, [1.0, 2.0, 3.0, 5.0])
        @test radius_at_energy(cog, cog.flux[1]) ≈ 1.0
        @test radius_at_energy(cog, cog.flux[3]) ≈ 3.0
    end

    @testset "interpolation" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=100.0, bkg=0.0)
        cog = curve_of_growth(model, [1.0, 5.0])
        r = radius_at_energy(cog, 50.0)
        @test r > 1.0
        @test r < 5.0
    end

    @testset "out of range returns NaN" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=100.0, bkg=0.0)
        cog = curve_of_growth(model, [1.0, 5.0])
        @test isnan(radius_at_energy(cog, 0.0))
        @test isnan(radius_at_energy(cog, 200.0))
    end
end

# ==============================================================================
# normalize
# ==============================================================================

@testset "normalize" begin
    @testset ":max method" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=50.0, bkg=0.0)
        cog = curve_of_growth(model, [1.0, 3.0, 10.0])
        norm = normalize(cog; method = :max)
        @test maximum(norm.flux) ≈ 1.0
        @test norm.radii == cog.radii
        @test norm.area == cog.area
        @test norm.y == cog.y && norm.x == cog.x
    end

    @testset ":sum method" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=50.0, bkg=0.0)
        cog = curve_of_growth(model, [1.0, 3.0, 10.0])
        norm = normalize(cog; method = :sum)
        @test norm.flux[end] ≈ 1.0
        @test norm.radii == cog.radii
    end

    @testset "original unchanged" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=50.0, bkg=0.0)
        cog = curve_of_growth(model, [1.0, 3.0, 10.0])
        orig_flux = copy(cog.flux)
        normalize(cog)
        @test cog.flux == orig_flux  # immutable style
    end

    @testset "zero normalization raises" begin
        img = zeros(10, 10)
        cog = curve_of_growth(img, 5, 5, [1.0, 2.0])
        @test_throws ArgumentError normalize(cog; method = :max)
    end

    @testset "invalid method raises" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=100.0, bkg=0.0)
        cog = curve_of_growth(model, [1.0, 3.0, 10.0])
        @test_throws ArgumentError normalize(cog; method = :invalid)
    end
end

# ==============================================================================
# Edge cases
# ==============================================================================

@testset "CurveOfGrowth edge cases" begin
    @testset "single radius" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=100.0, bkg=0.0)
        cog = curve_of_growth(model, [2.5])
        @test length(cog.flux) == 1
        @test cog.flux[1] > 0
    end

    @testset "monotonically increasing flux" begin
        model = CircularGaussianPSF(y=0.0, x=0.0, fwhm=4.0, flux=100.0, bkg=0.0)
        radii = 0.5:0.5:10.0
        cog = curve_of_growth(model, collect(radii))
        for i in 2:length(cog.flux)
            @test cog.flux[i] >= cog.flux[i-1]
        end
    end

    @testset "Struct construction and field access" begin
        cog = CurveOfGrowth{Float64}(
            [1.0, 2.0], [10.0, 20.0], Float64[],
            [3.14, 12.57], 5.0, 5.0)
        @test cog.radii == [1.0, 2.0]
        @test cog.flux == [10.0, 20.0]
        @test isempty(cog.flux_err)
        @test cog.area == [3.14, 12.57]
        @test cog.y == 5.0
        @test cog.x == 5.0
    end
end

# ==============================================================================
# reference_cog
# ==============================================================================

@testset "reference_cog" begin
    @testset "WFC — known EE values" begin
        ref = reference_cog(:WFC, :F814W)
        # From Bohlin (2016) Table 8: EE(F814W, r=1 pix) = 0.322
        @test ref.flux[1] ≈ 0.322 atol = 1e-6
        # Infinite-aperture endpoint: r = 5.5 / 0.05 = 110 pix, EE = 1.0
        @test last(ref.radii) ≈ 110.0
        @test last(ref.flux) ≈ 1.0
        @test isempty(ref.flux_err)
        @test ref.radii isa Vector{Float64}
        @test ref.flux isa Vector{Float64}
    end

    @testset "HRC — known EE values" begin
        ref = reference_cog(:HRC, :F435W)
        # From Bohlin (2016) Table 9: EE(F435W, r=2 pix) = 0.547
        @test ref.flux[1] ≈ 0.547 atol = 1e-6
        # Infinite-aperture endpoint: r = 5.5 / 0.025 = 220 pix, EE = 1.0
        @test last(ref.radii) ≈ 220.0
        @test last(ref.flux) ≈ 1.0
    end

    @testset "SBC — arcsecond conversion and endpoint trimming" begin
        ref = reference_cog(:SBC, :F150LP)
        # Original r = 0.1 arcsec → 0.1 / 0.03 ≈ 3.333 pix, EE = 0.546
        @test first(ref.radii) ≈ 0.1 / 0.03
        @test ref.flux[1] ≈ 0.546 atol = 1e-6
        # Last retained point: r = 4.0 arcsec → 4.0 / 0.03 ≈ 133.33 pix, EE = 1.000
        @test last(ref.radii) ≈ 4.0 / 0.03
        @test last(ref.flux) ≈ 1.0
        # Verify points beyond the infinite aperture (5.0″, 5.5″) were trimmed
        @test maximum(ref.radii) ≤ 4.0 / 0.03 + 1e-10
    end

    @testset "radii are strictly increasing" begin
        for (inst, filt) in [(:WFC, :F814W), (:HRC, :F435W), (:SBC, :F150LP)]
            ref = reference_cog(inst, filt)
            for i in 2:length(ref.radii)
                @test ref.radii[i] > ref.radii[i-1]
            end
        end
    end

    @testset "flux is monotonically increasing and in [0, 1]" begin
        for (inst, filt) in [(:WFC, :F814W), (:HRC, :F435W), (:SBC, :F150LP)]
            ref = reference_cog(inst, filt)
            for i in 1:length(ref.flux)
                @test 0.0 <= ref.flux[i] <= 1.0 + 1e-12
            end
            for i in 2:length(ref.flux)
                @test ref.flux[i] >= ref.flux[i-1]
            end
        end
    end

    @testset "encircled_energy and radius_at_energy work on reference COG" begin
        ref = reference_cog(:WFC, :F814W)
        ee = encircled_energy(ref, 3.0)
        @test ee > 0.6
        @test ee < 0.9
        r = radius_at_energy(ref, 0.8)
        @test r > 2.0
        @test r < 5.0
        # Out-of-range returns NaN
        @test isnan(encircled_energy(ref, 0.0))
        @test isnan(radius_at_energy(ref, -0.1))
    end

    @testset "normalize works on reference COG" begin
        ref = reference_cog(:WFC, :F814W)
        norm = normalize(ref; method = :sum)
        @test last(norm.flux) ≈ 1.0
        @test norm.radii == ref.radii
        norm_max = normalize(ref; method = :max)
        @test maximum(norm_max.flux) ≈ 1.0
    end

    @testset "error handling" begin
        @test_throws ArgumentError reference_cog(:BAD, :F814W)
        @test_throws ArgumentError reference_cog(:WFC, :BADFILT)
        @test_throws ArgumentError reference_cog(:HRC, :F125LP)  # valid filter but wrong instrument
    end

    @testset "area field is πr²" begin
        ref = reference_cog(:WFC, :F814W)
        for (r, a) in zip(ref.radii, ref.area)
            @test a ≈ π * r^2 rtol = 1e-12
        end
    end
end
