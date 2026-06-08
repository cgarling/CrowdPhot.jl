using CrowdPhot
using CrowdPhot.PSF
using Documenter
# using Documenter.Remotes: GitHub
using DocumenterCitations: CitationBibliography

setup = quote
    using CrowdPhot
    using CrowdPhot.PSF
end

DocMeta.setdocmeta!(CrowdPhot, :DocTestSetup, setup; recursive = true)

const CI = get(ENV, "CI", "false") == "true"
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

makedocs(;
    sitename = "CrowdPhot.jl",
    modules = [CrowdPhot, CrowdPhot.PSF],
    format = Documenter.HTML(;
        prettyurls = CI,
        assets = String[],
    ),
    authors = "Chris Garling",
    # repo = GitHub("cgarling/CrowdPhot.jl"),
    pages = [
        "Home" => "index.md",
        "PSFs" => [
            "PSF API" => "psf/psf_api.md",
            "Parametric PSF Models" => "psf/parametric_models.md",
            "Effective PSF Models" => [
                "ePSF Overview" => "psf/empirical/epsf_overview.md",
                "ImagePSF" => "psf/empirical/image_psf.md",
            ],
        ],
        "Levenberg-Marquardt Fitter" => "lm_fitter.md",
        "Simulation" => "simulation.md",
        "Index" => "doc_index.md",
        "References" => "refs.md",
        ],
    doctest = true,
    linkcheck = CI,
    warnonly = [:missing_docs, :linkcheck],
    plugins = [bib],
)

deploydocs(;
    repo = "github.com/cgarling/CrowdPhot.jl.git",
    devbranch = "main",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#"], # Restrict to minor releases
)