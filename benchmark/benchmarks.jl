# When adding a benchmark suite, update SUITE_NAMES and SUITE_TITLES, then
# define its BenchmarkGroup under SUITE using the same suite name.
using CrowdPhot: make_gaussians_image, simulate_image, centroid_poly, _centroid_poly3, correlate, findlocalmaxima, measure_star_shape, choose_centroid
using CrowdPhot.PSF: GaussianPSF, CircularGaussianPSF, CircularGaussianPRF, evaluate, evaluate_fg, fit_star, fit_psf, TukeyLoss, bicubic_interpolate, fill_grid_holes!, ImagePSF
using CrowdPhot.Background
import BackgroundMeshes as BM
using BenchmarkTools
using ImageFiltering: imfilter, findlocalmaxima as _if_findlocalmaxima
import LossFunctions
using OffsetArrays: centered
using PrettyTables: pretty_table
using StableRNGs: StableRNG

function show_benchmarks(results)
    # Collect results — results may be a flat Dict or a nested BenchmarkGroup;
    # flatten first so that every value is a Trial.
    flat = flatten_results(results)
    # Sort so that paired CrowdPhot / external-package benchmarks appear
    # adjacent, with the CrowdPhot entry first in each pair.
    sorted  = sort(collect(flat), by=pair -> begin
        key = pair.first
        base = replace(key, r"\s*,?\s*(?:BackgroundMeshes|ImageFiltering)\.jl" => "")
        ext  = occursin(r"(?:BackgroundMeshes|ImageFiltering)\.jl", key)
        return (base, ext)
    end)
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

function flatten_results(group)
    # Recursively flatten a (possibly nested) BenchmarkGroup of Trial results
    # into a flat Dict{String, Trial}.  Dict inputs are returned as-is so that
    # the function is idempotent.
    flat = Dict{String, Any}()
    _flatten_results!(flat, group, "")
    return flat
end

function _flatten_results!(flat, group, prefix)
    for (k, v) in group
        fullname = isempty(prefix) ? string(k) : "$prefix/$k"
        if v isa BenchmarkGroup
            _flatten_results!(flat, v, fullname)
        else
            flat[fullname] = v
        end
    end
end

const SUITE_NAMES = ["parametric", "fitting", "empirical", "background", "centroids", "correlation", "peakfinding", "morphology", "apertures"]
const SUITE_TITLES = Dict(
    "parametric" => "Parametric Suite",
    "fitting" => "Fitting Suite",
    "empirical" => "Empirical Suite",
    "background" => "Background Estimation Suite",
    "centroids" => "Centroids Suite",
    "correlation" => "Correlation Suite",
    "peakfinding" => "Peak Finding Suite",
    "morphology" => "Morphology Suite",
    "apertures" => "Aperture Suite",
)

function selected_suite_names(args)
    # Default to every suite when no command-line filters are provided.
    isempty(args) && return SUITE_NAMES

    # Fail early on misspelled suite names so benchmark output is unambiguous.
    unknown = setdiff(args, SUITE_NAMES)
    if !isempty(unknown)
        println(stderr, "Unknown benchmark suite(s): ", join(unknown, ", "))
        println(stderr, "Usage: julia benchmarks.jl [", join(SUITE_NAMES, "|"), "]...")
        exit(1)
    end

    # Preserve the user's requested order while avoiding duplicate runs.
    return unique(args)
end

function print_suite_header(name)
    # Format each selected suite consistently with the existing benchmark output.
    println("⎯⎯⎯ ", SUITE_TITLES[name], " ", repeat("⎯", 42))
end

function run_selected_suites(args)
    # Parse command-line filters once before executing any benchmark work.
    names = selected_suite_names(args)

    # Run and render each selected suite independently.
    for name in names
        results = run(SUITE[name], verbose=true)
        print_suite_header(name)
        show_benchmarks(results)
    end
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

for n in (5, 11, 21)
    SUITE["empirical"]["ImagePSF fit_star, size=($n, $n)"] = @benchmarkable fit_star(init, image, inds; max_iter = 100) setup=(begin
        origin = (y = ($n + 1) / 2, x = ($n + 1) / 2)
        grid_model = CircularGaussianPRF(x = origin.x, y = origin.y, fwhm = 2.4, flux = 1, bkg = 0)
        psf_data = evaluate.(grid_model, 1:$n, (1:$n)')
        truth = ImagePSF(psf_data; x = origin.x + 0.35, y = origin.y - 0.25,
            flux = 300.0, bkg = 4.0, origin, normalize = true)
        image = evaluate.(truth, 1:$n, (1:$n)')
        init = ImagePSF(psf_data; x = origin.x, y = origin.y + 0.1,
            flux = 260.0, bkg = 3.5, origin, normalize = true)
        inds = (1:$n, 1:$n)
    end)
end

for n in (50, 100)
    SUITE["empirical"]["ImagePSF fit_psf, n=$n"] = @benchmarkable psf, result =
        fit_psf(ImagePSF, image, sources.y, sources.x;
            psf_rad = 5.0, oversampling = 2, smooth = true, recenter = false,
            reweight = nothing) setup=(begin
                truth_model = CircularGaussianPRF(x = 0, y = 0, fwhm = 1.8, flux = 1, bkg = 0)
                image, sources = simulate_image((128, 128), truth_model, $n;
                    background = 20.0, noise = :none, flux = (600.0, 900.0),
                    min_separation = 7, border = 8, model_radius = 6)
            end) evals=1 samples=100
end

SUITE["background"] = BenchmarkGroup()
# Direct test of estimators
SUITE["background"]["estimators"] = BenchmarkGroup()
for t in (
    (MMMBackground(), BM.MMMBackground()), 
    (BiweightLocationBackground(), BM.BiweightLocationBackground()),
    (StdRMS(), BM.StdRMS()),
    (BiweightScaleRMS(), BM.BiweightScaleRMS()),
)
    img = make_gaussians_image(100, (100, 100); rng = StableRNG(7), background = 200.0, read_noise = 5.0, gain = 1.5)
    SUITE["background"]["estimators"]["$(typeof(t[1])) (100, 100)"] = @benchmarkable $t[1]($img)
    SUITE["background"]["estimators"]["$(typeof(t[1])) BackgroundMeshes.jl (100, 100)"] = @benchmarkable $t[2]($img)
end
# Test estimate_background on a range of image sizes, comparing against BackgroundMeshes.jl
for s in (30, 500, 2000)
    img = make_gaussians_image(s, (s, s); rng = StableRNG(7), background = 200.0, read_noise = 5.0, gain = 1.5)
    SUITE["background"]["MMMBackground, size=($s, $s)"] = @benchmarkable estimate_background($img; estimator = $MMMBackground(), rms_estimator = $StdRMS(), maxiters=$0)
    SUITE["background"]["MMMBackground, size=($s, $s), nclip=5"] = @benchmarkable estimate_background($img; estimator = $MMMBackground(), rms_estimator = $StdRMS(), maxiters=$5) # Test sigma clipping overhead
    SUITE["background"]["MMMBackground, BackgroundMeshes.jl, size=($s, $s)"] = @benchmarkable BM.estimate_background($img; location = $BM.MMMBackground(), rms = $BM.StdRMS())
end

let img = make_gaussians_image(1000, (1000, 1000); rng = StableRNG(7), background = 200.0, read_noise = 5.0, gain = 1.5)
    SUITE["background"]["Background2D (1000, 1000), 64 box, no filter"] = @benchmarkable Background2D($img, $64; estimator = MMMBackground(), rms_estimator = StdRMS(), maxiters=0)
    SUITE["background"]["Background2D BackgroundMeshes.jl (1000, 1000), 64 box, no filter"] = @benchmarkable BM.estimate_background($img, $64; location = $BM.MMMBackground(), rms = $BM.StdRMS())
end
# SUITE["background"]["Background2D (1000, 1000), BackgroundMeshes.jl, 64 box"] = @benchmarkable BM.Background2D($img, 64) setup=(img = make_gaussians_image(1000, (1000, 1000); rng = StableRNG(7), background = 200.0, read_noise = 5.0, gain = 1.5))
# SUITE["background"]["Background2D (1000, 1000), 64 box"] = @benchmarkable Background2D($img, 64) setup=(img = make_gaussians_image(1000, (1000, 1000); rng = StableRNG(7), background = 200.0, read_noise = 5.0, gain = 1.5))

# ---------------------------------------------------------------------------
# Centroids benchmarks
# ---------------------------------------------------------------------------
SUITE["centroids"] = BenchmarkGroup()

# Benchmark _centroid_poly3 directly on a 3×3 patch
let model = CircularGaussianPSF(x=2.0, y=2.0, fwhm=4.0, flux=10.0, bkg=0.0)
    inds = (1:3, 1:3)
    patch = evaluate.(model, inds[1], inds[2]')
    inv_var = ones(3, 3)
    SUITE["centroids"]["_centroid_poly3 (3x3)"] = @benchmarkable _centroid_poly3($patch, $inv_var)
end

# Benchmark centroid_poly on a larger cutout
for n in (7, 15)
    let model = CircularGaussianPSF(x=4.0, y=4.0, fwhm=4.0, flux=10.0, bkg=0.0)
        inds = (1:n, 1:n)
        img = evaluate.(model, inds[1], inds[2]')
        SUITE["centroids"]["centroid_poly ($n×$n)"] = @benchmarkable centroid_poly($img)
        cen = centroid_poly(img)
        SUITE["centroids"]["choose_centroid ($n×$n)"] = @benchmarkable choose_centroid($cen) samples=1000 evals=100
    end
end

# Benchmark centroid_poly on a larger cutout with explicit inverse variance
let model = CircularGaussianPSF(x=8.0, y=8.0, fwhm=4.0, flux=10.0, bkg=0.0)
    inds = (1:15, 1:15)
    img = evaluate.(model, inds[1], inds[2]')
    ivar = ones(15, 15)
    SUITE["centroids"]["centroid_poly (15×15), explicit ivar"] = @benchmarkable centroid_poly($img, $ivar)
end

# ---------------------------------------------------------------------------
# Correlation benchmarks — CrowdPhot.jl vs ImageFiltering.jl
# Note that ImageFiltering.jl has auto-algorithm selection; for small, separable
# kernels it uses direct, multi-threaded convolution `FIRTiled` with `CPUThreads`,
# while for large kernels it uses `FFT()`. For kernel sizes 5, 11 the direct convolution
# is used, while for 21 the FFT is used.
# ---------------------------------------------------------------------------
SUITE["correlation"] = BenchmarkGroup()

# Separable kernel: 2D Gaussian (always rank-1, SVD detection should fire).
function _bench_gaussian(σ, k)
    x = LinRange(-(k ÷ 2), k ÷ 2, k)
    g = exp.(-0.5 .* (x ./ σ) .^ 2)
    g ./= sum(g)
    return g * g'
end

for ksize in ((5, 5), (11, 11), (21, 21))
    for imsize in ((500, 500), (1000, 1000), (2000, 2000))
        # ---- separable (Gaussian) ----
        let k = ksize, sz = imsize
            σ = k[1] / 5
            label = "$sz, k=$k, separable"
            kern  = _bench_gaussian(σ, k[1])
            ckern = centered(kern)
            SUITE["correlation"]["$label, CrowdPhot.jl"] = @benchmarkable correlate(img, $(kern), :replicate) setup=(
                img = rand($sz...)
            ) samples=3
            SUITE["correlation"]["$label, ImageFiltering.jl"] = @benchmarkable imfilter(img, $(ckern), "replicate") setup=(
                img = rand($sz...)
            ) samples=3
        end
        # ---- non-separable (random full-rank) ----
        let k = ksize, sz = imsize
            label = "$sz, k=$k, non-separable"
            SUITE["correlation"]["$label, CrowdPhot.jl"] = @benchmarkable correlate(img, kern, :replicate) setup=(
                img = rand($sz...); kern = rand($k...)
            ) samples=3
            SUITE["correlation"]["$label, ImageFiltering.jl"] = @benchmarkable imfilter(img, ckern, "replicate") setup=(
                img = rand($sz...); kern = rand($k...); ckern = centered(kern)
            ) samples=3
        end
    end
end

# ---------------------------------------------------------------------------
# Peak-finding benchmarks — CrowdPhot.jl vs ImageFiltering.jl
# ---------------------------------------------------------------------------
SUITE["peakfinding"] = BenchmarkGroup()

for (sz, label) in [((100, 100), "100x100"), ((500, 500), "500x500"), ((2000, 2000), "2000x2000")]
    let s = sz, lbl = label
        SUITE["peakfinding"]["$lbl, CrowdPhot.jl"] = @benchmarkable findlocalmaxima(img) setup=(
            img = rand($s...)
        ) samples=10
        SUITE["peakfinding"]["$lbl, ImageFiltering.jl"] = @benchmarkable _if_findlocalmaxima(img; edges=true) setup=(
            img = rand($s...)
        ) samples=10
    end
end

# ---------------------------------------------------------------------------
# Morphology benchmarks — measure_star_shape on Gaussian cutouts
# ---------------------------------------------------------------------------
SUITE["morphology"] = BenchmarkGroup()

for n in (5, 11, 21)
    let sz = n
        model = CircularGaussianPSF(x = (sz + 1) / 2, y = (sz + 1) / 2,
            fwhm = 2.8, flux = 100.0, bkg = 0.0)
        inds = (1:sz, 1:sz)
        img = evaluate.(model, inds[1], inds[2]')
        SUITE["morphology"]["measure_star_shape ($sz×$sz)"] =
            @benchmarkable measure_star_shape($img)
    end
end

# ---------------------------------------------------------------------------
# Aperture benchmarks — CrowdPhot.jl vs Photometry.jl
# ---------------------------------------------------------------------------
SUITE["apertures"] = BenchmarkGroup()

using CrowdPhot: CircularAperture, bounding_axes, _overlap_flag, aperture_weight, ExactOverlap, CenterOverlap, WholePixelOverlap
import Photometry

# Set up equivalent apertures with coordinate swap.
# CrowdPhot: CircularAperture(y, x, r) — image-index convention.
# Photometry: CircularAperture(x, y, r) — matrix-index convention, ap[row=x, col=y].
const _CP_AP = CircularAperture(20.0, 20.0, 8.0)
const _PH_AP = Photometry.Aperture.CircularAperture(20.0, 20.0, 8.0)

# Collect pixel positions for per-pixel benchmarks.
# inside: pixel wholly inside the circle.
# partial: pixel crossing the boundary.
# outside: pixel wholly outside but in bounding box.
const _CP_YR, _CP_XR = bounding_axes(_CP_AP)
const _INSIDE_PIX  = (20, 20)       # center pixel
const _PARTIAL_PIX = (12, 20)       # near edge
const _OUTSIDE_PIX = (8, 8)         # well outside (but may be in box for r=8)

# Per-pixel weight evaluation — ExactOverlap
SUITE["apertures"]["ExactOverlap, inside pixel, CrowdPhot.jl"] =
    @benchmarkable aperture_weight($_CP_AP, 20, 20, ExactOverlap())
SUITE["apertures"]["ExactOverlap, inside pixel, Photometry.jl"] =
    @benchmarkable $_PH_AP[20, 20]

SUITE["apertures"]["ExactOverlap, partial pixel, CrowdPhot.jl"] =
    @benchmarkable aperture_weight($_CP_AP, 12, 20, ExactOverlap())
SUITE["apertures"]["ExactOverlap, partial pixel, Photometry.jl"] =
    @benchmarkable $_PH_AP[20, 12]  # coordinate swap: Photometry ap[row=x, col=y]

SUITE["apertures"]["ExactOverlap, outside pixel, CrowdPhot.jl"] =
    @benchmarkable aperture_weight($_CP_AP, 1, 1, ExactOverlap())
SUITE["apertures"]["ExactOverlap, outside pixel, Photometry.jl"] =
    @benchmarkable $_PH_AP[1, 1]

# Per-pixel weight evaluation — CenterOverlap
SUITE["apertures"]["CenterOverlap, CrowdPhot.jl"] =
    @benchmarkable aperture_weight($_CP_AP, 20, 20, CenterOverlap())
SUITE["apertures"]["CenterOverlap, manual"] =
    @benchmarkable ((20 - _CP_AP.x)^2 + (20 - _CP_AP.y)^2 < _CP_AP.r^2)

# Per-pixel weight evaluation — WholePixelOverlap
SUITE["apertures"]["WholePixelOverlap, inside pixel, CrowdPhot.jl"] =
    @benchmarkable aperture_weight($_CP_AP, 20, 20, WholePixelOverlap())
SUITE["apertures"]["WholePixelOverlap, partial pixel, CrowdPhot.jl"] =
    @benchmarkable aperture_weight($_CP_AP, 12, 20, WholePixelOverlap())

# _overlap_flag (fast-path discriminant)
SUITE["apertures"]["_overlap_flag, inside pixel, CrowdPhot.jl"] =
    @benchmarkable _overlap_flag($_CP_AP, 20, 20)
SUITE["apertures"]["overlap, inside pixel, Photometry.jl"] =
    @benchmarkable Photometry.Aperture.overlap($_PH_AP, 20, 20)

SUITE["apertures"]["_overlap_flag, partial pixel, CrowdPhot.jl"] =
    @benchmarkable _overlap_flag($_CP_AP, 12, 20)
SUITE["apertures"]["overlap, partial pixel, Photometry.jl"] =
    @benchmarkable Photometry.Aperture.overlap($_PH_AP, 20, 12)  # coord swap

# Bounding-box computation
SUITE["apertures"]["bounding_axes, CrowdPhot.jl"] =
    @benchmarkable bounding_axes($_CP_AP)
SUITE["apertures"]["bounds, Photometry.jl"] =
    @benchmarkable Photometry.Aperture.bounds($_PH_AP)

# Full-aperture weight sum — the curve-of-growth / aperture-photometry workhorse.
# CrowdPhot: loop over bounding_axes, call aperture_weight per pixel.
SUITE["apertures"]["full sum ExactOverlap, CrowdPhot.jl"] = @benchmarkable begin
    total = zero(Float64)
    for j in $_CP_XR, i in $_CP_YR
        w = aperture_weight($_CP_AP, i, j, ExactOverlap())
        total += w
    end
    total
end

# Photometry: iterate axes and sum getindex.  Fair comparison: both use lazy
# per-pixel evaluation, no allocation.
SUITE["apertures"]["full sum getindex, Photometry.jl"] = @benchmarkable begin
    total = zero(Float64)
    for idx in CartesianIndices(axes($_PH_AP))
        total += $_PH_AP[idx]
    end
    total
end

# Full-aperture binary mask sum (CenterOverlap) — fitting-region use case.
SUITE["apertures"]["full sum CenterOverlap, CrowdPhot.jl"] = @benchmarkable begin
    total = zero(Float64)
    for j in $_CP_XR, i in $_CP_YR
        w = aperture_weight($_CP_AP, i, j, CenterOverlap())
        total += w
    end
    total
end

# Manual inline distance check — best-case handwritten loop.
SUITE["apertures"]["full sum manual distance check"] = @benchmarkable begin
    total = zero(Float64)
    r2 = Float64(_CP_AP.r)^2
    yc = Float64(_CP_AP.y)
    xc = Float64(_CP_AP.x)
    for j in $_CP_XR, i in $_CP_YR
        if (Float64(i) - yc)^2 + (Float64(j) - xc)^2 < r2
            total += 1.0
        end
    end
    total
end

# If not on CI, we'll show a nice table
if get(ENV, "CI", "false") == "false"
    # Run the requested benchmarks and print a table for each suite.
    run_selected_suites(ARGS)
end
