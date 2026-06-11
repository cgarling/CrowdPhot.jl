# HST DRX

Here we will run photometry on an HST/ACS DRC file which 
is a high-level data product that has been calibrated,
geometrically-corrected, and dither-combined by AstroDrizzle.

```@example photometry
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
ax = Axis(fig[1,1]; aspect = DataAspect(), yreversed = true,
    xlabel = "x", ylabel = "y", 
    title="Leo A, HST F814W, Proposal ID 12273, PI: R. van der Marel",
    titlesize = 20)

hm = image!(ax, img; colorrange=zscale(img), colorscale=LuptonAsinhScale())
Colorbar(fig[1,2], hm; height = Relative(0.75), valign = :center)
fig
```