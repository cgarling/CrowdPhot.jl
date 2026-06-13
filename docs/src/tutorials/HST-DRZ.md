# HST DRX

Here we will run photometry on an HST/ACS DRC file which 
is a high-level data product that has been calibrated,
geometrically-corrected, and dither-combined by AstroDrizzle.

```@example hst-drc
using CrowdPhot
using FITSIO
using LazyArtifacts
using CairoMakie
using Makie: LuptonAsinhScale
using PlotUtils: zscale
using Pkg

# Load image from artifact system
artifacts_toml = joinpath(pkgdir(CrowdPhot), "docs", "Artifacts.toml")
artifact_dir = Pkg.Artifacts.ensure_artifact_installed("jbjl03010_drc", artifacts_toml)
fits_path = joinpath(artifact_dir, "jbjl03010_drc.fits")

# Load relevant parts from FITS file
hdr, img, weights = FITS(fits_path) do fits
    read_header(fits[1]), read(fits[2]), read(fits[3])
end

# Plot image
fig = Figure(size=(700,700),)
ax = Axis(fig[1,1]; aspect = DataAspect(),
    xlabel = "x", ylabel = "y", 
    title="Leo A, HST F814W, Proposal ID 12273, PI: R. van der Marel",
    titlesize = 20, yreversed = true)

colorrange = zscale(img) # Use PlotUtils.zscale for reasonable color limits
hm = image!(ax, img'; colorrange, colorscale=LuptonAsinhScale())
Colorbar(fig[1,2], hm; height = Relative(0.75), valign = :center)
fig
```

# Background Estimation
DRZ files have already been background-subtracted, but we will will still
run background estimation to get the noise (RMS) map.

```@example hst-drc
# `img` is NaN in areas not covered, allowing us to make a coverage mask
coverage_mask = isnan.(img)
# Estimate 2-D background with 256 pixel mesh size
bkg = Background2D(img, 256; coverage_mask, fill_value=NaN)
img_sub = img .- bkg.background

fig = Figure(size = (700, 300))
ax1 = Axis(fig[1, 1]; title = "Image", aspect = DataAspect(), yreversed = true)
ax2 = Axis(fig[1, 2]; title = "Background model", aspect = DataAspect(), yreversed = true)
ax3 = Axis(fig[1, 3]; title = "Residual", aspect = DataAspect(), yreversed = true)

im1 = image!(ax1, img'; colorrange)
im2 = image!(ax2, bkg.background'; colorrange)
im3 = image!(ax3, img_sub'; colorrange)
# Colorbar(fig[2, 3], hm; vertical = false)
for (i, im) in enumerate((im1, im2, im3))
    Colorbar(fig[2, i], im; vertical = false)
end
for ax in (ax1, ax2, ax3)
    hidedecorations!(ax)
end

fig
```

## Source Detection

We detect point sources with [`matched_filter`](@ref) using a circular
Gaussian kernel with FWHM = 2.0 pixels, the approximate width of the
ACS/WFC PSF at F814W.  We provide the inverse-variance map from the
background RMS so that noisier regions are properly down-weighted.

```@example hst-drc
# Convert to Float64 for consistency with the PSF kernel
img_sub_f64 = Float64.(img_sub)

# Inverse variance from background RMS; clip NaN regions to zero weight.
inv_var = fill(0.0, size(img_sub_f64))
valid = @. !isnan(bkg.background_rms) & (bkg.background_rms > 0)
@. inv_var[valid] = 1 / bkg.background_rms[valid]^2

# PSF FWHM for HST/ACS F814W (~2 pix)
psf_fwhm = 2.0
mf = matched_filter(img_sub_f64, psf_fwhm; inv_var, sigma = 5.0)

println("Detected $(length(mf.peaks)) sources at ≥ 5σ")
```

And now we can plot our detections on a small part of the image:

```@example hst-drc
region_width = 500
ny, nx = size(img_sub)
y_start = 1500
x_start = 1500
y_range = y_start:min(ny, y_start + region_width - 1)
x_range = x_start:min(nx, x_start + region_width - 1)

region_img = img_sub[y_range, x_range]
region_colorrange = zscale(region_img[isfinite.(region_img)])

# CrowdPhot coordinates are image[y, x]; Makie overlays take (x, y).
region_peaks = filter(mf.peaks) do peak
    y, x = Tuple(peak)
    y in y_range && x in x_range
end

# Draw 4-pixel-diameter source circles in image-coordinate units.
source_circles = map(region_peaks) do peak
    y, x = Tuple(peak)
    Circle(Point2f(x, y), 2)
end

fig = Figure(size = (700, 700))
ax = Axis(fig[1, 1];
    aspect = DataAspect(), yreversed = true,
    xlabel = "x", ylabel = "y",
    title = "Detected sources in a 500×500 pixel region")

hm = heatmap!(ax, x_range, y_range, region_img';
    colorrange = region_colorrange, colorscale = LuptonAsinhScale(),
    colormap = :grays, interpolate = false)
poly!(ax, source_circles; color = (:limegreen, 0), strokecolor = :limegreen,
    strokewidth = 1.2)
Colorbar(fig[1, 2], hm; height = Relative(0.75), valign = :center)

fig
```

## Morphological Measurements

We feed the [`MatchedFilterResult`](@ref) directly to
[`measure_star_shapes`](@ref), which extracts a cutout around each peak,
computes sub-pixel centroids via [`centroid_poly`](@ref), and measures
aperture-based FWHM, roundness, and flux via
[`measure_star_shape`](@ref).

Moment-based statistics can be biased if the cutout includes many
background-dominated pixels. Here we use a `half_width=2` so the full
width of the cutout is 4 pixels, twice the FWHM of the kernel we used
for detection, which gives good results.

```@example hst-drc
# Measure morphology for all detected sources
results = measure_star_shapes(mf; half_width = 2)

# Compute instrumental magnitudes from aperture flux
inst_mag = Float64[]
for r in results
    flux = r.morphology.flux
    push!(inst_mag, flux > 0 ? -2.5 * log10(flux) : NaN)
end

# Filter to sources with valid measurements
good = findall(results) do r
    r.morphology.fwhm.y > 0 &&
    r.morphology.fwhm.x > 0 &&
    isfinite(r.core.normalized_curvature) &&
    r.morphology.flux > 0
end

println("$(length(good)) / $(length(results)) sources have valid morphological measurements")
```

## Morphology Diagnostics

The panels below show how morphological measurements trend with
instrumental magnitude.  Brighter sources (lower instrumental magnitude)
have more reliable shape measurements, while fainter sources scatter
more due to noise.  We use 2-D histograms to visualize density in the
crowded regions. For this HST example, the "core" morphology statistics
(which are calculated on the 3x3 pixels surrounding the centroid) will
be about equivalent to the aperture statistics because our aperture
box size is only 5x5 pixels, so they do not add much information in this
case.

```@example hst-drc
using Statistics

# Extract measurements for valid sources
mags = inst_mag[good]
fwhm_y  = [results[i].morphology.fwhm.y for i in good]
fwhm_x  = [results[i].morphology.fwhm.x for i in good]
fwhm_theta = [results[i].morphology.fwhm.theta for i in good]
round1 = [results[i].morphology.roundness1_aperture for i in good]
round2 = [results[i].morphology.roundness2_aperture for i in good]
round1_core = [results[i].core.roundness1_core for i in good]
round2_core = [results[i].core.roundness2_core for i in good]
sharp  = [results[i].core.normalized_curvature for i in good]
sig    = [results[i].significance for i in good]
mf_flux = [results[i].matched_filter_flux for i in good]

fig = Figure(size = (900, 1000))

# Panel 1: FWHM y vs magnitude
ax1 = Axis(fig[1, 1]; xlabel = "Instrumental magnitude",
           ylabel = "FWHM y (pix)", title = "FWHM (y-axis)")
h1 = hexbin!(ax1, mags, fwhm_y; bins = 80)
Colorbar(fig[1, 2], h1; label = "Counts")

# Panel 2: FWHM x vs magnitude
ax2 = Axis(fig[1, 3]; xlabel = "Instrumental magnitude",
           ylabel = "FWHM x (pix)", title = "FWHM (x-axis)")
h2 = hexbin!(ax2, mags, fwhm_x; bins = 80)
Colorbar(fig[1, 4], h2; label = "Counts")

# Panel 3: roundness1 (SROUND) vs magnitude
ax3 = Axis(fig[2, 1]; xlabel = "Instrumental magnitude",
           ylabel = "roundness1", title = "roundness1_aperture (SROUND)")
h3 = hexbin!(ax3, mags, round1; bins = 80)
Colorbar(fig[2, 2], h3; label = "Counts")

# Panel 4: roundness2 (GROUND) vs magnitude
ax4 = Axis(fig[2, 3]; xlabel = "Instrumental magnitude",
           ylabel = "roundness2", title = "roundness2_aperture (GROUND)")
h4 = hexbin!(ax4, mags, round2; bins = 80)
Colorbar(fig[2, 4], h4; label = "Counts")

# Panel 5: normalized curvature vs magnitude
ax5 = Axis(fig[3, 1]; xlabel = "Instrumental magnitude",
           ylabel = "normalized curvature", title = "Normalized Core Curvature (2/FWHM²) / I_0")
ylims!(ax5, -1, 3)
idxs = findall(x -> -10 < x < 10, sharp)
h5 = hexbin!(ax5, mags[idxs], sharp[idxs]; bins = 80, colorscale=log10)
Colorbar(fig[3, 2], h5; label = "Counts")

# Panel 6: median FWHM per magnitude bin
mag_bins = range(minimum(mags), maximum(mags); length = 30)
med_fwhm_y = Float64[]
med_fwhm_x = Float64[]
mag_centers = Float64[]
for (lo, hi) in zip(mag_bins[1:end-1], mag_bins[2:end])
    bin_idx = findall(m -> lo <= m < hi, mags)
    isempty(bin_idx) && continue
    push!(mag_centers, (lo + hi) / 2)
    push!(med_fwhm_y, median(fwhm_y[bin_idx]))
    push!(med_fwhm_x, median(fwhm_x[bin_idx]))
end
ax6 = Axis(fig[3, 3]; xlabel = "Instrumental magnitude",
           ylabel = "Median FWHM (pix)", title = "Median FWHM by magnitude")
scatterlines!(ax6, mag_centers, med_fwhm_y; label = "y", color = :blue)
scatterlines!(ax6, mag_centers, med_fwhm_x; label = "x", color = :red)
axislegend(ax6; position = :rt)

# Panel 7: Core SROUND vs Aperture SROUND
ax7 = Axis(fig[4, 1]; xlabel = "roundness1 aperture",
           ylabel = "roundness1 core", title = "Core vs Aperture roundness1 (SROUND)",
           limits = ((-2, 2), (-2, 2)))
h7 = hexbin!(ax7, round1, round1_core; bins = 80, colorscale=log10)
Colorbar(fig[4, 2], h7; label = "Counts")

# Panel 8: Core GROUND vs Aperture GROUND
ax8 = Axis(fig[4, 3]; xlabel = "roundness2 aperture",
           ylabel = "roundness2 core", title = "Core vs Aperture roundness2 (GROUND)",
           limits = ((-2, 2), (-2, 2)))
h8 = hexbin!(ax8, round2, round2_core; bins = 80, colorscale=log10)
Colorbar(fig[4, 4], h8; label = "Counts")


fig
```
