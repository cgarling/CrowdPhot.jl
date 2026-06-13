using CrowdPhot: measure_star_shape, _moments2, matched_filter,
    MatchedFilterResult, centroid_poly, choose_centroid, measure_star_shapes
using CrowdPhot.PSF: CircularGaussianPSF, GaussianPSF, evaluate, fwhm as psf_fwhm
using FillArrays: Fill
using StableRNGs: StableRNG
using Test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _make_gaussian_cutout(; x0=5.0, y0=5.0, fwhm=2.8, flux=100.0, shape=(9,9))
    inds = (1:shape[1], 1:shape[2])
    model = CircularGaussianPSF(x=x0, y=y0, fwhm=fwhm, flux=flux, bkg=0.0)
    img = evaluate.(model, inds[1], inds[2]')
    return img, model
end

function _make_elliptical_gaussian(; x0=5.0, y0=5.0, x_fwhm=3.0, y_fwhm=1.5, theta=30.0, flux=100.0, shape=(11,11))
    inds = (1:shape[1], 1:shape[2])
    model = GaussianPSF(x=x0, y=y0, x_fwhm=x_fwhm, y_fwhm=y_fwhm, theta=theta, flux=flux, bkg=0.0)
    img = evaluate.(model, inds[1], inds[2]')
    return img, model
end

# ---------------------------------------------------------------------------
# _moments2
# ---------------------------------------------------------------------------

@testset "_moments2" begin

    @testset "noise-free Gaussian — positive moments" begin
        img, _ = _make_gaussian_cutout()
        inv_var = Fill(1.0, size(img))
        mom = _moments2(img, inv_var, 0.0, 5.0, 5.0)
        @test mom.M00 > 0
        @test abs(mom.M10) < 1.0    # centroid near reference point
        @test abs(mom.M01) < 1.0
        @test mom.M20 > 0
        @test mom.M02 > 0
        @test mom.M11 > -1e-6       # circular => near zero
    end

    @testset "zero background — all flux captured" begin
        img, _ = _make_gaussian_cutout(; flux=200.0, fwhm=2.0)
        mom = _moments2(img, Fill(1.0, size(img)), 0.0, 5.0, 5.0)
        @test mom.M00 ≈ 200.0 rtol=0.01
    end

    @testset "background subtraction" begin
        img, _ = _make_gaussian_cutout(; flux=100.0, fwhm=2.0)
        # All pixels > 0, so bg=0 and bg=50 should give different M00
        mom0 = _moments2(img, Fill(1.0, size(img)), 0.0, 5.0, 5.0)
        mom50 = _moments2(img, Fill(1.0, size(img)), 50.0, 5.0, 5.0)
        @test mom0.M00 > mom50.M00
    end

    @testset "zero-weight pixels are skipped" begin
        img, _ = _make_gaussian_cutout()
        w = ones(size(img))
        w[5, 5] = 0.0  # mask the peak
        mom_masked = _moments2(img, w, 0.0, 5.0, 5.0)
        mom_full = _moments2(img, Fill(1.0, size(img)), 0.0, 5.0, 5.0)
        @test mom_masked.M00 < mom_full.M00
    end

    @testset "all pixels below background — M00 = 0" begin
        img = fill(1.0, 5, 5)
        mom = _moments2(img, Fill(1.0, size(img)), 10.0, 3.0, 3.0)
        @test mom.M00 == 0.0
    end

    @testset "single pixel" begin
        img = zeros(5, 5)
        img[3, 3] = 10.0
        mom = _moments2(img, Fill(1.0, size(img)), 0.0, 3.0, 3.0)
        @test mom.M00 == 10.0
        @test mom.M10 == 0.0
        @test mom.M01 == 0.0
        @test mom.M20 == 0.0
        @test mom.M02 == 0.0
        @test mom.M11 == 0.0
    end
end

# ---------------------------------------------------------------------------
# measure_star_shape — core method
# ---------------------------------------------------------------------------

@testset "measure_star_shape" begin

    @testset "noise-free circular Gaussian" begin
        # Large cutout centered on the star to avoid edge truncation bias.
        img, model = _make_gaussian_cutout(; x0=8.0, y0=8.0, fwhm=2.8, shape=(17,17))
        result = measure_star_shape(img, 8, 8; background=0)
        y_fwhm_model, x_fwhm_model = psf_fwhm(model)
        @test result.fwhm.y ≈ y_fwhm_model rtol=0.03
        @test result.fwhm.x ≈ x_fwhm_model rtol=0.03
        # theta is undefined for a perfectly circular PSF; just check finite.
        @test isfinite(result.fwhm.theta)
        @test abs(result.roundness1_aperture) < 0.30   # SROUND ~0 for symmetric
        @test abs(result.roundness2_aperture) < 0.15   # GROUND ~0 for circular
        @test result.flux > 0
        @test result.centroid.y ≈ 8.0 atol=0.5
        @test result.centroid.x ≈ 8.0 atol=0.5
    end

    @testset "noise-free elliptical Gaussian" begin
        # Large cutout so Gaussian wings are not truncated.
        img, model = _make_elliptical_gaussian(;
            x0=10.0, y0=10.0, x_fwhm=3.0, y_fwhm=1.5, theta=30.0, shape=(21,21))
        result = measure_star_shape(img, 10, 10; background=0)
        # y_fwhm < x_fwhm for this model
        @test result.fwhm.y < result.fwhm.x
        # Moment-based FWHM approximates the Gaussian FWHM; tolerances
        # are loose because finite-aperture moments differ from the
        # analytic infinite-integral FWHM.
        @test result.fwhm.y ≈ 1.5 rtol=0.35
        @test result.fwhm.x ≈ 3.0 rtol=0.35
        @test abs(result.fwhm.theta - 30.0) < 15.0
        # Extended in x → GROUND negative.
        @test result.roundness2_aperture < -0.1
    end

    @testset "constant image — NaN" begin
        img = fill(5.0, 7, 7)
        result = measure_star_shape(img, 4, 4; background=10)
        @test isnan(result.fwhm.y)
        @test isnan(result.fwhm.x)
        @test isnan(result.fwhm.theta)
        @test isnan(result.roundness2_aperture)
        @test isnan(result.centroid.y)
        @test isnan(result.centroid.x)
    end

    @testset "single bright pixel — near-zero width" begin
        img = zeros(7, 7)
        img[4, 4] = 100.0
        result = measure_star_shape(img, 4, 4; background=0)
        # Single pixel has zero spatial extent → FWHM NaN.
        @test isnan(result.fwhm.y)
        @test isnan(result.fwhm.x)
        # Zero-width → degenerate (denominator vanishes), treated as isotropic.
        @test result.roundness2_aperture == 0.0
        @test result.flux == 100.0
        @test result.centroid.y ≈ 4.0
        @test result.centroid.x ≈ 4.0
    end

    @testset "zero-weight pixels" begin
        img, _ = _make_gaussian_cutout()
        w = ones(size(img))
        w[5, 5] = 0.0
        result_full = measure_star_shape(img, 5, 5)
        result_masked = measure_star_shape(img, 5, 5; inv_var=w)
        @test result_masked.flux < result_full.flux
    end

    @testset "sub-pixel centroid via measure_star_shape convenience" begin
        img, _ = _make_gaussian_cutout(; x0=5.2, y0=5.3, shape=(15,15))
        result = measure_star_shape(img)
        # Convenience method finds the peak then calls the core.
        @test result.fwhm.y > 0
        @test result.fwhm.x > 0
        @test abs(result.roundness1_aperture) < 0.30
        @test abs(result.roundness2_aperture) < 0.2  # ~0 for circular
    end

    @testset "Float32 precision" begin
        img_f32 = Float32[0.1 0.3 0.1; 0.3 1.0 0.3; 0.1 0.3 0.1]
        result = measure_star_shape(img_f32; background=0)
        @test result.fwhm.y > 0
        @test result.fwhm.x > 0
        @test result.roundness2_aperture isa Float32
        @test result.flux isa Float32
    end

    @testset "shift invariance — integer pixel" begin
        # A shifted cutout of the same star should give the same FWHM.
        img1, _ = _make_gaussian_cutout(; x0=5.0, y0=5.0, shape=(15,15))
        img2, _ = _make_gaussian_cutout(; x0=7.0, y0=7.0, shape=(15,15))
        r1 = measure_star_shape(img1, 5, 5)
        r2 = measure_star_shape(img2, 7, 7)
        @test r1.fwhm.y ≈ r2.fwhm.y rtol=0.01
        @test r1.fwhm.x ≈ r2.fwhm.x rtol=0.01
    end

    @testset "theta is finite for circular PSF" begin
        img, _ = _make_gaussian_cutout(; x0=10.0, y0=10.0, shape=(21,21))
        result = measure_star_shape(img, 10, 10)
        @test isfinite(result.fwhm.theta)
        @test abs(result.roundness2_aperture) < 0.15   # ~0 for circular
    end

    @testset "cosmic ray — sharp star comparison" begin
        # A cosmic ray (single bright pixel) should have extreme sharpness
        # from centroid_poly.  Here we test that a normal star has moderate
        # sharpness and the single-pixel case is handled.
        img_star, _ = _make_gaussian_cutout(; fwhm=2.8, shape=(9,9))
        img_cr = zeros(9, 9)
        img_cr[5, 5] = 100.0

        # measure_star_shape reports NaN FWHM for single pixel (zero width)
        r_star = measure_star_shape(img_star, 5, 5; background=0)
        r_cr = measure_star_shape(img_cr, 5, 5; background=0)
        @test r_star.fwhm.y > 0
        @test isnan(r_cr.fwhm.y)   # zero-width → NaN
    end

    @testset "centroid covariance" begin
        img, _ = _make_gaussian_cutout(; x0=8.0, y0=8.0, shape=(17,17))
        result = measure_star_shape(img, 8, 8; background=0)
        @test result.centroid.y_err > 0
        @test result.centroid.x_err > 0
        @test result.centroid.cov[1,1] ≈ result.centroid.y_err^2
        @test result.centroid.cov[2,2] ≈ result.centroid.x_err^2
        @test result.centroid.cov[1,2] ≈ result.centroid.cov[2,1]
    end

    @testset "asymmetric — one-sided feature (SROUND)" begin
        # Star with a bright pixel on one side should give nonzero SROUND.
        img, _ = _make_gaussian_cutout(; x0=5.0, y0=5.0, fwhm=2.0, shape=(9,9))
        r_sym = measure_star_shape(img, 5, 5; background=0)
        @test r_sym.roundness1_aperture ≈ 0 atol=1e-10  # SROUND ~0 for symmetric
        # Add a diffraction-spike-like feature to the right side.
        img[5, 7] += 50.0
        img[5, 8] += 30.0
        r_asym = measure_star_shape(img, 5, 5; background=0)
        # Right-side feature: dy=0, dx>0 → quad1 → sign -1 → SROUND negative.
        @test r_asym.roundness1_aperture < -0.05
    end

    @testset "SROUND/GROUND divergence — symmetric opposite-side pair" begin
        # Flux on the same diagonal (top-left + bottom-right) keeps
        # M20 ≈ M02 so GROUND stays ~0, but both corners are quad3/quad1
        # with sign −1 in SROUND, so SROUND becomes negative.  This is
        # where the two statistics provide complementary information.
        img, _ = _make_gaussian_cutout(; x0=5.0, y0=5.0, fwhm=2.0, shape=(9,9))
        img[3, 3] += 80.0  # top-left
        img[7, 7] += 80.0  # bottom-right
        r = measure_star_shape(img, 5, 5; background=0)
        @test abs(r.roundness2_aperture) ≈ 0 atol = 1e-10  # GROUND ~0 (σ² balanced)
        @test r.roundness1_aperture < 0 # SROUND negative
    end

    @testset "asymmetric elliptical Gaussian (SROUND and GROUND)" begin
        # Axis-aligned ellipse: SROUND and GROUND have the same sign
        # because the ellipticity produces both bilateral asymmetry and
        # unequal marginal heights.  They diverge for rotated or
        # non-elliptical features (e.g. one-sided diffraction spikes).
        #
        # Extended in x → both negative.
        img, _ = _make_elliptical_gaussian(; x_fwhm=4.0, y_fwhm=2.0, theta=0.0,
            x0=10.0, y0=10.0, shape=(21,21))
        r = measure_star_shape(img, 10, 10; background=0)
        @test r.roundness1_aperture < -0.5  # SROUND: x-elongation
        @test r.roundness2_aperture < -0.5   # GROUND: HX < HY
        @test r.fwhm.x > r.fwhm.y

        # Extended in y → both positive.
        img2, _ = _make_elliptical_gaussian(; x_fwhm=2.0, y_fwhm=4.0, theta=0.0,
            x0=10.0, y0=10.0, shape=(21,21))
        r2 = measure_star_shape(img2, 10, 10; background=0)
        @test r2.roundness1_aperture > 0.5
        @test r2.roundness2_aperture > 0.5
        @test r2.fwhm.y > r2.fwhm.x
    end

    @testset "roundness sign agreement (core vs aperture)" begin
        # Both roundness fields should have the same sign.
        using CrowdPhot: centroid_poly
        img, _ = _make_elliptical_gaussian(; x_fwhm=3.0, y_fwhm=1.5, theta=0.0,
            x0=10.0, y0=10.0, shape=(21,21))
        cent = centroid_poly(img)
        shape = measure_star_shape(img, 10, 10; background=0)
        @test sign(cent.roundness1_core) == sign(shape.roundness1_aperture)
        @test sign(cent.roundness2_core) == sign(shape.roundness2_aperture)
    end

    @testset "broad elliptical PSF — core vs aperture roundness" begin
        using CrowdPhot: centroid_poly
        # Elliptical Gaussian with broad FWHM: both core and aperture
        # detect the ellipticity.  The 3×3 curvature measurement is
        # actually more sensitive than the moment-based aperture because
        # curvature at the peak falls off faster along the narrow axis.
        model = GaussianPSF(x=16.0, y=16.0, x_fwhm=6.0, y_fwhm=3.0,
            theta=0.0, flux=1000.0, bkg=0.0)
        img = evaluate.(model, 1:31, (1:31)')
        cent = centroid_poly(img)
        shape = measure_star_shape(img, 16, 16; background=0)
        @test cent.roundness2_core < -0.5        # x-extended
        @test shape.roundness2_aperture < -0.5   # x-extended
        @test cent.roundness2_core < -0.5  # core detects strong ellipticity
        @test shape.fwhm.x > shape.fwhm.y
    end

    @testset "background exclusion — z>0 in _moments2" begin
        img, _ = _make_gaussian_cutout(; x0=5.0, y0=5.0, flux=200.0, fwhm=2.0, shape=(9,9))
        # With bg=1000, all pixels are below background → M00=0.
        r_below = measure_star_shape(img, 5, 5; background=1000)
        @test r_below.flux == 0.0
        @test isnan(r_below.fwhm.y)
        # With bg=5, only the central pixels exceed background.
        r_partial = measure_star_shape(img, 5, 5; background=5)
        @test r_partial.flux > 0
        @test r_partial.flux < 200.0  # less than total star flux
    end

    @testset "noisy image — roundness bounded" begin
        using StableRNGs: StableRNG
        rng = StableRNG(42)
        img, _ = _make_gaussian_cutout(; x0=3.5, y0=3.5, fwhm=2.8, flux=200.0, shape=(7,7))
        noisy = img .+ 5.0 .* randn(rng, size(img))
        r = measure_star_shape(noisy, 4, 4; background=0)
        @test isfinite(r.roundness1_aperture)
        @test isfinite(r.roundness2_aperture)
        @test r.fwhm.y > 0
        @test r.fwhm.x > 0
        @test r.centroid.y_err > 0
        @test r.centroid.x_err > 0
    end
end

# ---------------------------------------------------------------------------
# measure_star_shapes — batch measurement from MatchedFilterResult
# ---------------------------------------------------------------------------

@testset "measure_star_shapes" begin
    rng = StableRNG(99)

    @testset "noise-free circular Gaussian" begin
        # Place a single source in a clean image, detect and measure.
        x0, y0 = 25.3, 25.7
        fwhm_val = 3.0
        flux_val = 500.0
        model = CircularGaussianPSF(; x=x0, y=y0, fwhm=fwhm_val, flux=flux_val, bkg=0.0)
        img = evaluate.(model, 1:50, (1:50)')
        mf = matched_filter(img, fwhm_val; sigma=4.0)
        @test length(mf.peaks) == 1

        results = measure_star_shapes(mf)
        @test length(results) == 1
        r = results[1]

        # Peak index and pixel
        @test r.peak_index == 1
        @test r.pixel == mf.peaks[1]

        # Centroid near true position
        @test abs(r.centroid.y - y0) < 1.0
        @test abs(r.centroid.x - x0) < 1.0

        # FWHM approximately recovered
        y_fwhm_true, x_fwhm_true = psf_fwhm(model)
        @test r.morphology.fwhm.y ≈ y_fwhm_true rtol=0.20
        @test r.morphology.fwhm.x ≈ x_fwhm_true rtol=0.20

        # Roundness near zero for circular PSF
        @test abs(r.core.roundness1_core) < 0.30
        @test abs(r.core.roundness2_core) < 0.30
        @test abs(r.morphology.roundness1_aperture) < 0.30
        @test abs(r.morphology.roundness2_aperture) < 0.30

        # Core and aperture roundness are both near zero for a circular PSF.
        # Sign comparison is meaningless when values are ~0; test magnitude.
        @test abs(r.core.roundness1_core) < 0.30
        @test abs(r.core.roundness2_core) < 0.30

        # Flux is positive
        @test r.morphology.flux > 0
        @test r.significance > 0
        @test r.matched_filter_flux > 0

        # Centroid errors are positive
        @test r.core.poly.y_err > 0
        @test r.core.poly.x_err > 0
        @test r.morphology.centroid.y_err > 0
        @test r.morphology.centroid.x_err > 0
    end

    @testset "noise-free elliptical Gaussian" begin
        # Elliptical PSF with known orientation.
        x0, y0 = 30.0, 30.0
        x_fwhm, y_fwhm = 4.0, 2.0
        theta = 30.0
        model = GaussianPSF(; x=x0, y=y0, x_fwhm, y_fwhm, theta, flux=500.0, bkg=0.0)
        img = evaluate.(model, 1:60, (1:60)')
        mf = matched_filter(img, (y_fwhm, x_fwhm); sigma=4.0)
        @test length(mf.peaks) == 1

        results = measure_star_shapes(mf)
        r = results[1]

        # y_FWHM < x_FWHM for this model
        @test r.morphology.fwhm.y < r.morphology.fwhm.x

        # Extended in x → GROUND negative
        @test r.core.roundness2_core < -0.1
        @test r.morphology.roundness2_aperture < -0.1

        # SROUND sign on the 3×3 core can differ from aperture SROUND
        # for rotated PSFs; test aperture SROUND for the expected sign.
        @test r.morphology.roundness1_aperture < -0.1
    end

    @testset "empty peaks (high sigma)" begin
        # Pure noise, high threshold → no detections.
        img = randn(rng, 50, 50)
        mf = matched_filter(img, 3.0; sigma=10.0)
        @test isempty(mf.peaks)

        results = measure_star_shapes(mf)
        @test results isa Vector
        @test isempty(results)
    end

    @testset "peaks keyword — subset selection" begin
        # Two sources, measure only the first.
        model1 = CircularGaussianPSF(; x=15.0, y=15.0, fwhm=2.5, flux=300.0, bkg=0.0)
        model2 = CircularGaussianPSF(; x=35.0, y=35.0, fwhm=2.5, flux=200.0, bkg=0.0)
        img = evaluate.(model1, 1:50, (1:50)') .+ evaluate.(model2, 1:50, (1:50)')
        mf = matched_filter(img, 3.0; sigma=4.0)
        @test length(mf.peaks) >= 2

        # Measure only the first peak.
        results = measure_star_shapes(mf; peaks=[1])
        @test length(results) == 1
        @test results[1].peak_index == 1

        # Measure only the second peak.
        results2 = measure_star_shapes(mf; peaks=[2])
        @test length(results2) == 1
        @test results2[1].peak_index == 2
    end

    @testset "min_significance filtering" begin
        # Two sources with different fluxes → different significances.
        model_bright = CircularGaussianPSF(; x=15.0, y=15.0, fwhm=2.5, flux=500.0, bkg=0.0)
        model_faint  = CircularGaussianPSF(; x=35.0, y=35.0, fwhm=2.5, flux=50.0, bkg=0.0)
        img = evaluate.(model_bright, 1:50, (1:50)') .+
              evaluate.(model_faint, 1:50, (1:50)') .+
              randn(rng, 50, 50) .* 0.5
        inv_var = fill(4.0, size(img))
        mf = matched_filter(img, 3.0; inv_var, sigma=3.0)

        # Bright source should have much higher significance than faint.
        sig_bright = maximum(mf.peak_significances)
        # Filter to only the brightest.
        results = measure_star_shapes(mf; min_significance=sig_bright * 0.8)
        @test length(results) >= 1
        # All returned peaks should satisfy the threshold.
        for r in results
            @test r.significance >= sig_bright * 0.8
        end
    end

    @testset "clipped cutout from large half_width" begin
        # Place a source near the image corner and use a large half_width
        # so the cutout is clipped on two sides by the image boundary.
        model = CircularGaussianPSF(; x=5.0, y=5.0, fwhm=2.5, flux=500.0, bkg=0.0)
        img = evaluate.(model, 1:50, (1:50)')
        mf = matched_filter(img, 3.0; sigma=4.0)
        @test length(mf.peaks) >= 1

        # Use a half_width larger than the distance to the image corner,
        # forcing clipping on the top and left edges.
        results = measure_star_shapes(mf; half_width=10)
        r = results[1]

        # The 3×3 core is complete since the peak is at least 2 px inside,
        # so centroid_poly should succeed.
        @test isfinite(r.core.poly.y)
        @test isfinite(r.core.poly.x)
        @test r.morphology.flux > 0
        @test r.centroid.y > 0
        @test r.centroid.x > 0
    end

    @testset "inv_var=nothing (uniform) vs explicit inv_var" begin
        x0, y0 = 25.0, 25.0
        model = CircularGaussianPSF(; x=x0, y=y0, fwhm=3.0, flux=400.0, bkg=0.0)
        img = evaluate.(model, 1:50, (1:50)')

        # With explicit inv_var
        mf_ivar = matched_filter(img, 3.0; inv_var=fill(1.0, size(img)), sigma=4.0)
        r_ivar = measure_star_shapes(mf_ivar)[1]

        # Without inv_var (uniform assumed)
        mf_none = matched_filter(img, 3.0; sigma=4.0)
        r_none = measure_star_shapes(mf_none)[1]

        # Both should find the source at the same position.
        @test r_ivar.peak_index == r_none.peak_index
        # Morphology fluxes should be similar (both use uniform weights
        # since inv_var=1.0 is uniform).
        @test r_ivar.morphology.flux ≈ r_none.morphology.flux rtol=0.01
    end

    @testset "coordinate consistency" begin
        # Verify that all global coordinates are consistent: the centroid
        # from core, chosen centroid, and morphology centroid should all
        # be near each other.
        x0, y0 = 25.3, 25.7
        model = CircularGaussianPSF(; x=x0, y=y0, fwhm=3.0, flux=500.0, bkg=0.0)
        img = evaluate.(model, 1:50, (1:50)')
        mf = matched_filter(img, 3.0; sigma=4.0)
        results = measure_star_shapes(mf)
        r = results[1]

        # All three centroid estimates should be near each other.
        @test abs(r.core.poly.y - r.centroid.y) < 1.0
        @test abs(r.core.poly.x - r.centroid.x) < 1.0
        @test abs(r.morphology.centroid.y - r.centroid.y) < 1.0
        @test abs(r.morphology.centroid.x - r.centroid.x) < 1.0

        # All should be near the true position.
        @test abs(r.centroid.y - y0) < 1.0
        @test abs(r.centroid.x - x0) < 1.0
    end

    @testset "custom half_width" begin
        x0, y0 = 30.0, 30.0
        model = CircularGaussianPSF(; x=x0, y=y0, fwhm=3.0, flux=500.0, bkg=0.0)
        img = evaluate.(model, 1:60, (1:60)')
        mf = matched_filter(img, 3.0; sigma=4.0)

        # Small half_width — cutout is tight around the peak.
        r_small = measure_star_shapes(mf; half_width=3)[1]
        # Large half_width — cutout captures more of the PSF wings.
        r_large = measure_star_shapes(mf; half_width=10)[1]

        # Larger cutout captures more flux.
        @test r_large.morphology.flux > r_small.morphology.flux
        # Both should have reasonable FWHM estimates.
        @test r_small.morphology.fwhm.y > 0
        @test r_large.morphology.fwhm.y > 0
    end

    @testset "noisy image — multiple sources" begin
        # Multiple sources with noise: verify we get one result per peak.
        img = randn(rng, 50, 50) .* 0.3
        σ_psf = 3.0 / 2.355
        kern_half = ceil(Int, 4 * σ_psf) |> k -> isodd(k) ? k : k + 1
        x = LinRange(-kern_half÷2, kern_half÷2, kern_half)
        g = exp.(-0.5 .* (x ./ σ_psf) .^ 2)
        g ./= sum(g)
        kernel = g .* g'

        # Place three sources.
        for (xx, yy, flux) in [(15.0, 15.0, 80.0), (35.0, 20.0, 60.0), (25.0, 40.0, 40.0)]
            kr = kern_half ÷ 2
            xr = max(1, round(Int, xx - kr)):min(50, round(Int, xx + kr))
            yr = max(1, round(Int, yy - kr)):min(50, round(Int, yy + kr))
            for j in xr, i in yr
                dx, dy = j - xx, i - yy
                img[i, j] += flux * exp(-(dx^2 + dy^2) / (2σ_psf^2)) / (2π * σ_psf^2)
            end
        end

        inv_var = fill(1 / 0.3^2, size(img))
        mf = matched_filter(img, kernel; inv_var, sigma=4.0)
        results = measure_star_shapes(mf)

        @test length(results) == length(mf.peaks)
        @test length(results) >= 2  # at least 2 of 3 detected at 4σ

        # Each result should have valid morphology.
        for r in results
            @test r.peak_index >= 1
            @test r.morphology.flux > 0
            @test r.morphology.fwhm.y > 0
            @test r.morphology.fwhm.x > 0
            @test isfinite(r.core.normalized_curvature)
        end
    end
end
