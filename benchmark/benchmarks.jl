using CrowdPhot: simulate_image
using CrowdPhot.PSF: GaussianPSF, CircularGaussianPSF, CircularGaussianPRF, evaluate, evaluate_fg, fit_star, fit_psf, TukeyLoss, bicubic_interpolate, fill_grid_holes!, ImagePSF
using BenchmarkTools
import LossFunctions
using PrettyTables: pretty_table

function show_benchmarks(results)
    # Collect results
    sorted  = sort(collect(results), by=first)
    names   = [k for (k,_) in sorted]
    trials  = [v for (_,v) in sorted]

    # Pack into matrix
    data = hcat(
        names,
        [BenchmarkTools.prettytime(median(t).time) for t in trials],
        [BenchmarkTools.prettymemory(median(t).memory) for t in trials],
        [median(t).allocs for t in trials]
    )

    # Make pretty table
    pretty_table(data;
        column_labels = ["Benchmark", "Median Time", "Memory", "Allocs"],
        alignment     = [:l, :r, :r, :r]
    )
end

const SUITE = BenchmarkGroup()
SUITE["parametric"] = BenchmarkGroup()

let model = CircularGaussianPSF(x=15.0, y=15.0, fwhm=4.0, flux=10.0, bkg=1.0)
    idx = CartesianIndex(15, 15)
    SUITE["parametric"]["circular_gaussian_evaluate_fg"] = @benchmarkable evaluate_fg($model, $idx)
end

# ---------------------------------------------------------------------------
# LM fitting benchmarks
# ---------------------------------------------------------------------------
SUITE["fitting"] = BenchmarkGroup()

let model = CircularGaussianPSF(x=15.0, y=15.0, fwhm=4.0, flux=10.0, bkg=1.0)
    inds  = (1:30, 1:30)
    image = evaluate.(model, inds[1], inds[2]')
    init = CircularGaussianPSF(x=15.5, y=14.5, fwhm=3.5, flux=9.0, bkg=1.2)
    SUITE["fitting"]["fit_star CircularGaussianPSF (L2)"] = @benchmarkable fit_star($init, $image, $inds)
    SUITE["fitting"]["fit_star CircularGaussianPSF (Huber IRLS)"] = @benchmarkable fit_star($init, $image, $inds;
        reweight=$(LossFunctions.HuberLoss(1.0)))
    SUITE["fitting"]["fit_star CircularGaussianPSF (Tukey IRLS)"] = @benchmarkable fit_star($init, $image, $inds;
        reweight=$(TukeyLoss()))
end

# ---------------------------------------------------------------------------
# Empirical model benchmarks
# ---------------------------------------------------------------------------
SUITE["empirical"] = BenchmarkGroup()
SUITE["empirical"]["bicubic_interpolate"] = @benchmarkable bicubic_interpolate(x, $3.5, $3.5) setup=(x=rand(7,7))
SUITE["empirical"]["fill_grid_holes!"] = @benchmarkable fill_grid_holes!(x) setup=(x=rand(21,21); inds=([9, 4, 15, 13, 1, 1], [6, 15, 17, 1, 17, 1]); x[inds...] .= NaN) evals=1

for n in (50, 100)
    SUITE["empirical"]["ImagePSF fit_psf, n=$n"] = @benchmarkable psf, result =
        fit_psf(ImagePSF, image, sources.x, sources.y;
            psf_rad = 5.0, oversampling = 2, smooth = true, recenter = false,
            reweight = nothing) setup=(begin
                truth_model = CircularGaussianPRF(x = 0, y = 0, fwhm = 1.8, flux = 1, bkg = 0)
                image, sources = simulate_image((128, 128), truth_model, $n;
                    background = 20.0, noise = :none, flux = (600.0, 900.0),
                    min_separation = 7, border = 8, model_radius = 6)
            end) evals=1
end

# If not on CI, we'll show a nice table
if get(ENV, "CI", "false") == "false"
    # Run the benchmarks
    results = run(SUITE, verbose=true)
    println("⎯⎯⎯ Parametric Suite ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯")
    show_benchmarks(results["parametric"])
    println("⎯⎯⎯ Fitting Suite ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯")
    show_benchmarks(results["fitting"])
    println("⎯⎯⎯ Empirical Suite ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯")
    show_benchmarks(results["empirical"])
end
