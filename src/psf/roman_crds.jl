"""
    roman_crds_gridded_epsf(path; defocus=0, spectral_type="G2V", psf_subtype="psf",
                            origin=nothing, normalize=false) -> GriddedPSFModel

Read a Roman CRDS ePSF reference file (ASDF) into a `GriddedPSFModel` of
`ImagePSF` nodes. Requires `ASDF.jl` to be loaded (`using ASDF`); this
function is implemented in a package extension (`CrowdPhotASDFExt`).

The oversampling factor is always read from the file's own
`meta.oversample`; it is a fact about how the file's PSF stamps were
tabulated, not a user choice, so it is not a keyword argument here.

`defocus` selects the defocus-waves slice (`0` = in-focus, matching
`romanisim`'s default). `spectral_type` selects the spectral-type slice by
name (default `"G2V"`, matching `romanisim`'s default index). `psf_subtype`
selects between `"psf"` (default, includes interpixel-capacitance) and
`"psf_noipc"`; `"extended_psf"`/`"extended_psf_noipc"` (single non-gridded
stamps) are not supported here.

If the reference file's PSF stamps are the optical-PSF-only convention
(not yet convolved with the detector pixel response, i.e. summing to
approximately 1 rather than approximately `oversample^2`), this function
convolves each node with the detector pixel-response function before use,
matching `romancal`'s `get_gridded_psf_model`. See the package
documentation for the full normalization discussion.
"""
function roman_crds_gridded_epsf end
