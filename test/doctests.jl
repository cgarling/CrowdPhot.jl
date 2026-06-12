# Run doctests

import CrowdPhot
import Documenter: DocMeta, doctest
DocMeta.setdocmeta!(CrowdPhot, :DocTestSetup, :(using CrowdPhot); recursive=true)
doctest(CrowdPhot)