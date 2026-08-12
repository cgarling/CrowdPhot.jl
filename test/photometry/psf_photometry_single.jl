using CrowdPhot
using CrowdPhot: CircularGaussianPSF, _clamp_inds
using CrowdPhot.PSF: free_params
using StableRNGs
using Statistics: median
using Test

# ---------------------------------------------------------------------------
# _clamp_inds
# ---------------------------------------------------------------------------

@testset "_clamp_inds" begin
    img = zeros(10, 10)

    @testset "fully inside" begin
        inds = CartesianIndices((3:7, 3:7))
        clamped = _clamp_inds(inds, img)
        @test clamped == inds
        @test length(clamped) == 25
    end

    @testset "partially outside" begin
        inds = CartesianIndices((8:12, 8:12))
        clamped = _clamp_inds(inds, img)
        @test clamped == CartesianIndices((8:10, 8:10))
        @test length(clamped) == 9
    end

    @testset "completely outside" begin
        inds = CartesianIndices((11:15, 3:7))
        clamped = _clamp_inds(inds, img)
        @test length(clamped) == 0
    end
end

# ---------------------------------------------------------------------------
# _extract_source_catalog
# ---------------------------------------------------------------------------

@testset "_extract_source_catalog" begin
    psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)

    @testset "NamedTuple with :y, :x" begin
        sources = (; y=[10.0, 20.0], x=[30.0, 40.0], flux=[500.0, 600.0])
        params, errors = CrowdPhot._extract_source_catalog(sources, psf, Float64)
        @test size(params) == (5, 2)
        @test params[1, :] == [10.0, 20.0]  # y
        @test params[2, :] == [30.0, 40.0]  # x
        @test params[4, :] == [500.0, 600.0] # flux
        @test params[5, :] == [0.0, 0.0]     # bkg (default)
    end

    @testset "NamedTuple without optional fields" begin
        sources = (; y=[5.0], x=[15.0])
        params, errors = CrowdPhot._extract_source_catalog(sources, psf, Float64)
        @test params[4, 1] == 1.0  # flux default
        @test params[5, 1] == 0.0  # bkg default
    end

    @testset "NamedTuple with bkg" begin
        sources = (; y=[5.0], x=[15.0], bkg=[3.0])
        params, _ = CrowdPhot._extract_source_catalog(sources, psf, Float64)
        @test params[5, 1] == 3.0
    end

    @testset "fwhm row filled from psf default" begin
        sources = (; y=[5.0], x=[15.0])
        params, _ = CrowdPhot._extract_source_catalog(sources, psf, Float64)
        @test params[3, 1] == 2.0  # fwhm from psf
    end

    @testset "errors initialized to NaN" begin
        sources = (; y=[5.0], x=[15.0])
        _, errors = CrowdPhot._extract_source_catalog(sources, psf, Float64)
        @test all(isnan, errors)
    end
end

# ---------------------------------------------------------------------------
# _extract_errors!
# ---------------------------------------------------------------------------

@testset "_extract_errors!" begin
    T = Float64
    errors = fill(T(NaN), 3, 1)
    cov = T[4.0 0.5; 0.5 1.0]
    free_idx = (1, 3)
    is_fixed = (2,)

    CrowdPhot._extract_errors!(errors, cov, free_idx, is_fixed, 1)
    @test errors[1, 1] ≈ 2.0         # sqrt(4.0)
    @test errors[2, 1] == 0.0         # fixed
    @test errors[3, 1] ≈ 1.0         # sqrt(1.0)
end

# ---------------------------------------------------------------------------
# fit_all_stars — integration
# ---------------------------------------------------------------------------

@testset "fit_all_stars" begin
    rng = StableRNG(42)
    T = Float64
    truth_psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)

    @testset "noiseless recovery" begin
        image, sources = simulate_image((128, 128), truth_psf, 5;
            background = 20.0, noise = :none, flux = (600.0, 900.0),
            min_separation = 7, border = 8, model_radius = 30, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        # Sequentially fitting blended stars leaves residual crosstalk even
        # with multiple passes; DAOPHOT handles this via simultaneous group
        # fits.  Tolerances reflect what the sequential algorithm can achieve.
        result = fit_all_stars(image, psf, sources, 5; n_passes = 3, max_iter = 100, fixed=(; bkg = 20.0))

        @test result.n_passes == 3
        @test result.n_failed == 0
        @test isempty(result.failure_msgs)
        @test sum(result.valid) == 5
        @test all(result.converged)
        for i in 1:5
            @test result.flux[i] ≈ sources.flux[i] rtol = 0.10
            @test result.y[i] ≈ sources.y[i] atol = 0.01
            @test result.x[i] ≈ sources.x[i] atol = 0.01
            @test isfinite(result.qfit[i])
            @test result.qfit[i] > 0
        end
        # qfit_expected and qfit_z are NaN when inv_var is not provided.
        @test all(isnan, result.qfit_expected)
        @test all(isnan, result.qfit_z)
        # Crowding: finite, non-negative (may be positive due to blending).
        @test all(isfinite, result.crowding)
        @test all(x -> x >= 0, result.crowding)
    end

    @testset "single-pass runs without error" begin
        image, sources = simulate_image((128, 128), truth_psf, 5;
            background = 20.0, noise = :none, flux = (600.0, 900.0),
            min_separation = 10, border = 10, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        inv_var = fill(1.0, size(image))
        result = fit_all_stars(image, psf, sources, 5; n_passes = 1, max_iter = 100, inv_var)
        @test result.n_passes == 1
        @test all(result.valid)
        @test all(result.converged)
        @test all(isfinite, result.qfit_expected)
        @test all(x -> x > 0, result.qfit_expected)
        @test all(isfinite, result.qfit_z)
    end

    @testset "fixed background" begin
        image, sources = simulate_image((64, 64), truth_psf, 5;
            background = 20.0, noise = :none, flux = (500.0, 800.0),
            min_separation = 10, border = 8, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        result = fit_all_stars(image, psf, sources, 5;
            fixed = (; bkg = 20.0), n_passes = 1, max_iter = 100)

        @test all(x -> x ≈ 20.0, result.bkg)
        @test all(iszero, result.bkg_err)
        for i in 1:5
            @test result.flux[i] ≈ sources.flux[i] rtol = 0.02
        end
    end

    @testset "fixed positions" begin
        image, sources = simulate_image((64, 64), truth_psf, 5;
            background = 20.0, noise = :none, flux = (500.0, 800.0),
            min_separation = 10, border = 8, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        result = fit_all_stars(image, psf, sources, 5;
            fixed = (; x = 32.0, y = 32.0), n_passes = 1, max_iter = 100)

        @test all(x -> x ≈ 32.0, result.x)
        @test all(y -> y ≈ 32.0, result.y)
        @test all(iszero, result.x_err)
        @test all(iszero, result.y_err)
        # Fixed-position errors are zero; flux and bkg errors may be NaN
        # (no covariance_estimator) or near-zero (noiseless fit).
        @test all(isfinite, result.y_err)
        @test all(isfinite, result.x_err)
    end

    @testset "edge stars" begin
        image, _ = simulate_image((64, 64), truth_psf, 2;
            background = 20.0, noise = :none, flux = (500.0, 600.0),
            min_separation = 30, border = 30, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        # One star at the edge, one at center
        sources = (; y = [1.0, 32.0], x = [32.0, 32.0], flux = [500.0, 500.0])
        result = fit_all_stars(image, psf, sources, 5; n_passes = 1, max_iter = 50)
        # Edge star may or may not survive depending on cutout size
        @test result.valid[2]  # center star should be valid
    end

    @testset "multi-pass does not error" begin
        image, sources = simulate_image((64, 64), truth_psf, 5;
            background = 20.0, noise = :none, flux = (400.0, 700.0),
            min_separation = 5, border = 8, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        result = fit_all_stars(image, psf, sources, 5; n_passes = 3, max_iter = 100)
        @test result.n_passes == 3
        @test sum(result.valid) == 5
        @test all(result.converged)
    end

    @testset "NamedTuple input with fluxes" begin
        image, sources = simulate_image((64, 64), truth_psf, 3;
            background = 20.0, noise = :none, flux = (500.0, 700.0),
            min_separation = 15, border = 10, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        # Use the actual source positions from the simulation, with reasonable
        # initial flux guesses.
        cat = (; y = sources.y, x = sources.x, flux = fill(400.0, 3))
        result = fit_all_stars(image, psf, cat, 5; n_passes = 1, max_iter = 200)
        @test length(result.y) == 3
        @test all(result.converged)
        for i in 1:3
            @test result.flux[i] ≈ sources.flux[i] rtol = 0.05
        end
    end

    @testset "return struct fields have correct types" begin
        image, sources = simulate_image((64, 64), truth_psf, 3;
            background = 20.0, noise = :none, flux = (500.0, 700.0),
            min_separation = 15, border = 10, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        result = fit_all_stars(image, psf, sources, 5; n_passes = 1, max_iter = 100)

        @test result.y isa Vector{Float64}
        @test result.x isa Vector{Float64}
        @test result.flux isa Vector{Float64}
        @test result.bkg isa Vector{Float64}
        @test result.y_err isa Vector{Float64}
        @test result.flux_err isa Vector{Float64}
        @test result.converged isa BitVector
        @test result.valid isa BitVector
        @test result.finalized isa BitVector
        @test result.n_iter isa Vector{Int}
        @test result.qfit isa Vector{Float64}
        @test result.qfit_expected isa Vector{Float64}
        @test result.qfit_z isa Vector{Float64}
        @test result.crowding isa Vector{Float64}
        @test result.n_failed isa Int
        @test result.failure_msgs isa Vector{String}
        @test result.residual isa Matrix{Float64}
        @test size(result.residual) == size(image)
    end

    @testset "invalid flux rejected between passes" begin
        image, sources = simulate_image((64, 64), truth_psf, 5;
            background = 20.0, noise = :none, flux = (400.0, 700.0),
            min_separation = 10, border = 8, model_radius = 5, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        # Include a source with negative flux that should be rejected
        sources_bad = (; y = [sources.y; 50.0],
                         x = [sources.x; 50.0],
                         flux = [sources.flux; -100.0])
        result = fit_all_stars(image, psf, sources_bad, 5; n_passes = 2, max_iter = 100)
        @test sum(result.valid) == 5  # 5 good + 1 bad rejected
        @test !result.valid[6]
    end
end

# ---------------------------------------------------------------------------
# fit_all_stars — finalize step for bright stars
# ---------------------------------------------------------------------------

@testset "fit_all_stars — finalize step" begin
    T = Float64

    @testset "closed-form estimator is exact for noiseless data, any footprint" begin
        # With y, x, bkg fixed at their (exactly recovered) truth values, the
        # closed-form finalize sum reproduces the injected flux to high
        # precision regardless of how large the finalize footprint is
        # (num/den simplifies to the true flux identically in the noiseless
        # limit). A small amount of noise is retained so that the LM
        # covariance (and hence the finalize-selection SNR) is well-defined
        # -- a truly noiseless fit has an exactly-zero residual variance, for
        # which "SNR" is not a meaningful concept.
        rng = StableRNG(1)
        y0, x0, flux0, bkg0 = 40.0, 40.0, 5000.0, 20.0
        wide_psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=8.0, flux=1.0, bkg=0.0)
        image = simulate_image((81, 81), wide_psf, (; y=[y0], x=[x0], flux=[flux0]);
            background = bkg0, noise = :gaussian, read_noise = 0.5, model_radius = 35, rng)
        sources = (; y=[y0], x=[x0], flux=[flux0])

        result = fit_all_stars(image, wide_psf, sources, 5;
            n_passes = 1, max_iter = 200, fixed = (; bkg = bkg0),
            finalize_snr_min = 1.0, finalize_rad = 30)

        @test result.finalized[1]
        @test result.flux[1] ≈ flux0 rtol = 1e-3
        @test isfinite(result.flux_err[1])
    end

    @testset "variance reduction from a larger footprint" begin
        # A star with wide (fwhm=8) wings fit only over a small fit_rad=5
        # core throws away information available at larger radii; the
        # finalize step should recover a smaller flux uncertainty than the
        # small-footprint fit alone.
        y0, x0, flux0, bkg0 = 50.0, 50.0, 20000.0, 20.0
        wide_psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=8.0, flux=1.0, bkg=0.0)
        sources = (; y=[y0], x=[x0], flux=[flux0])

        image_a = simulate_image((101, 101), wide_psf, sources;
            background = bkg0, noise = :gaussian, read_noise = 5.0, model_radius = 45, rng = StableRNG(2))
        image_b = copy(image_a)

        result_small = fit_all_stars(image_a, wide_psf, sources, 5;
            n_passes = 1, max_iter = 200, fixed = (; bkg = bkg0))
        result_large = fit_all_stars(image_b, wide_psf, sources, 5;
            n_passes = 1, max_iter = 200, fixed = (; bkg = bkg0),
            finalize_snr_min = 1.0, finalize_rad = 30)

        @test !result_small.finalized[1]
        @test result_large.finalized[1]
        @test isfinite(result_small.flux_err[1])
        @test isfinite(result_large.flux_err[1])
        @test result_large.flux_err[1] < result_small.flux_err[1]

        # qfit/qfit_expected/qfit_z are scoped to the small `inds` footprint
        # regardless of finalize status, so they should be identical between
        # the two runs (the small-footprint LM fit itself is unaffected by
        # whether finalize triggers afterward).
        @test result_small.qfit[1] ≈ result_large.qfit[1]
    end

    @testset "finalize decision is fixed at pass 1" begin
        rng = StableRNG(3)
        truth_psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=3.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((80, 80), truth_psf, 4;
            background = 20.0, noise = :gaussian, read_noise = 3.0,
            flux = (500.0, 15000.0), min_separation = 12, border = 10,
            model_radius = 15, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=3.0, flux=1.0, bkg=0.0)

        result = fit_all_stars(image, psf, sources, 5;
            n_passes = 3, max_iter = 100, fixed = (; bkg = 20.0),
            finalize_snr_min = 20.0, finalize_rad = 15)

        # `finalized` reflects the pass-1 SNR test only; re-running with
        # n_passes = 1 (i.e. stopping right after the same pass-1 fits that
        # decide membership) must select exactly the same stars.
        result_pass1_only = fit_all_stars(image, psf, sources, 5;
            n_passes = 1, max_iter = 100, fixed = (; bkg = 20.0),
            finalize_snr_min = 20.0, finalize_rad = 15)
        @test result.finalized == result_pass1_only.finalized
    end

    @testset "finalize_rad validation" begin
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        sources = (; y=[10.0], x=[10.0], flux=[100.0])
        image = fill(20.0, 32, 32)
        @test_throws ArgumentError fit_all_stars(image, psf, sources, 5; finalize_rad = 2)
    end

    @testset "backward compatibility: finalize disabled by default" begin
        rng = StableRNG(4)
        truth_psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((64, 64), truth_psf, 3;
            background = 20.0, noise = :none, flux = (500.0, 900.0),
            min_separation = 12, border = 10, model_radius = 10, rng)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        result = fit_all_stars(image, psf, sources, 5; n_passes = 2, max_iter = 100)
        @test all(x -> x == false, result.finalized)
    end

    @testset "warnings: fixed flux disables finalize; infinite-support model warns" begin
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        sources = (; y=[16.0], x=[16.0], flux=[1000.0])
        image = fill(20.0, 32, 32)

        # `flux` fixed: finalize cannot compute an SNR, should warn once and
        # disable finalize (no error, `finalized` stays all-false).
        @test_logs (:warn,) match_mode = :any result_fixed_flux = fit_all_stars(
            image, psf, sources, 5; n_passes = 1, fixed = (; flux = 1000.0),
            finalize_snr_min = 1.0,
        )

        # Analytic model, `finalize_rad === nothing`: should warn about
        # falling back to the model's default (5xFWHM) extent.
        @test_logs (:warn,) match_mode = :any fit_all_stars(
            image, psf, sources, 5; n_passes = 1, finalize_snr_min = 1.0,
        )
    end

    @testset "crowding tracks the finalize footprint; qfit stays core-scoped" begin
        # Bright star A with a faint neighbor B sitting between fit_rad and
        # finalize_rad: invisible to A's small `inds` aperture (so baseline
        # crowding ~ 0) but inside A's finalize footprint (so finalize-scope
        # crowding should detect the contamination).
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=3.0, flux=1.0, bkg=0.0)
        y_a, x_a, flux_a = 40.0, 40.0, 30000.0
        y_b, x_b, flux_b = 40.0, 51.0, 500.0
        bkg0 = 20.0
        sources = (; y=[y_a, y_b], x=[x_a, x_b], flux=[flux_a, flux_b])
        image = simulate_image((80, 80), psf, sources;
            background = bkg0, noise = :none, model_radius = 20)

        result_baseline = fit_all_stars(image, psf, sources, 5;
            n_passes = 3, max_iter = 100, fixed = (; bkg = bkg0))
        result_finalized = fit_all_stars(image, psf, sources, 5;
            n_passes = 3, max_iter = 100, fixed = (; bkg = bkg0),
            finalize_snr_min = 10.0, finalize_rad = 15)

        @test !result_baseline.finalized[1]
        @test result_finalized.finalized[1]
        @test !result_finalized.finalized[2]  # faint neighbor stays below threshold

        # qfit stays scoped to the small core aperture regardless of finalize
        # status, so it should be small (good fit) in both configurations;
        # exact equality is not expected here since finalize's large-footprint
        # subtraction (which overlaps the faint neighbor B) changes what add_star!
        # restores across passes, which can perturb A's LM convergence path (not
        # its converged value) at the floating-point/tolerance level. This is
        # already checked more directly (fixed data, isolated star) in the
        # "variance reduction" testset above.
        @test result_baseline.qfit[1] < 1e-3
        @test result_finalized.qfit[1] < 1e-3

        # crowding for the finalized star should detect B's contamination
        # (positive, larger than the non-finalized baseline which cannot see
        # B at all from within the small aperture).
        @test result_finalized.crowding[1] > result_baseline.crowding[1]
    end
end
