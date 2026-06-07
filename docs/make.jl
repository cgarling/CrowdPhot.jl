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
            # "PSF Overview" => "psf_api.md",
            # "Parametric PSF Models" => "parametric_models.md",
            "Effective PSF Models" => 
                [
            "ePSF Overview" => "empirical/epsf_overview.md",
            "ImagePSF" => "empirical/image_psf.md"
            ],
        ],
        # "Benchmarks" => "bench.md",
    ],
    doctest = false,
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