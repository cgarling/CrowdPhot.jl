# PSF-fitting photometry for a single image.
#
# Implements a DOLPHOT-style multi-pass algorithm: stars are sorted by
# brightness and fitted one at a time against a residual image from which
# all brighter (and, on later passes, all other) stars have been subtracted.
# On passes 2+, each star is added back before re-fitting so it sees the
# original data minus only its neighbours.

# ==============================================================================
# Result type
# ==============================================================================

"""
    MultiPassPhotResult{T}

Result of [`fit_all_stars`](@ref).  All per-star vectors have the same
length (the number of input sources).

# Fields

- `y`, `x`: fitted centroid positions.
- `y_err`, `x_err`: 1-sigma centroid errors (`zero(T)` if positions were fixed).
- `flux`: fitted flux.
- `flux_err`: 1-sigma flux error (`zero(T)` if flux was fixed).
- `bkg`: fitted local background.
- `bkg_err`: 1-sigma background error (`zero(T)` if background was fixed).
- `converged::BitVector`: whether the final LM fit converged.
- `valid::BitVector`: whether the source survived between-pass validation.
- `chisq`: final reduced χ² for each source.
- `n_passes::Int`: number of passes actually performed.
- `residual::Matrix{T}`: final residual image after all subtractions.
"""
struct MultiPassPhotResult{T}
    y::Vector{T}
    x::Vector{T}
    y_err::Vector{T}
    x_err::Vector{T}
    flux::Vector{T}
    flux_err::Vector{T}
    bkg::Vector{T}
    bkg_err::Vector{T}
    converged::BitVector
    valid::BitVector
    chisq::Vector{T}
    n_passes::Int
    residual::Matrix{T}
end

# ==============================================================================
# Source catalog extraction
# ==============================================================================

# Shared helper: build the internal params matrix from per-star initial values.
# `overrides` is a NamedTuple mapping field-name → Vector of per-star values
# for the fields we want to set from the source catalog.  Rows for other PSF
# fields are filled from the PSF model defaults and are identical across stars.
function _build_params_matrix(psf, overrides::NamedTuple, T::Type)
    defaults = ConstructionBase.getproperties(psf)
    prop_names = collect(keys(defaults))
    n_params = length(prop_names)
    n_stars = length(first(values(overrides)))
    params = Matrix{T}(undef, n_params, n_stars)
    errors = Matrix{T}(undef, n_params, n_stars)
    for (k, name) in enumerate(prop_names)
        vals = if haskey(overrides, name)
            T.(getfield(overrides, name))
        else
            fill(T(getfield(defaults, name)), n_stars)
        end
        params[k, :] .= vals
        errors[k, :] .= T(NaN)
    end
    return params, errors
end

"""
    _extract_source_catalog(sources::Vector{<:NamedTuple}, psf, T)

Extract initial source parameters from the output of
[`measure_star_shapes`](@ref).  Uses `centroid.y`, `centroid.x`,
`matched_filter_flux` (or `morphology.aperture_sum`) as the initial
guesses; `bkg` defaults to zero.
"""
function _extract_source_catalog(sources::Vector{<:NamedTuple}, psf, ::Type{T}) where {T}
    n = length(sources)
    y  = T[s.centroid.y for s in sources]
    x  = T[s.centroid.x for s in sources]
    flux = T[s.matched_filter_flux for s in sources]
    bkg = zeros(T, n)
    return _build_params_matrix(psf, (; y, x, flux, bkg), T)
end

"""
    _extract_source_catalog(sources::NamedTuple, psf, T)

Extract initial source parameters from a `NamedTuple` with fields `:y` and
`:x` (required).  Optional fields `:flux` (defaults to 1) and `:bkg`
(defaults to 0).
"""
function _extract_source_catalog(sources::NamedTuple, psf, ::Type{T}) where {T}
    n = length(sources.y)
    y  = T.(sources.y)
    x  = T.(sources.x)
    flux = haskey(sources, :flux) ? T.(sources.flux) : fill(T(1), n)
    bkg  = haskey(sources, :bkg)  ? T.(sources.bkg)  : zeros(T, n)
    return _build_params_matrix(psf, (; y, x, flux, bkg), T)
end

"""
    _extract_source_catalog(mf::MatchedFilterResult, psf, T)

Extract initial source parameters from a [`MatchedFilterResult`](@ref).
Uses `Tuple.(peaks)` for pixel-centre `(y, x)`, `peak_fluxes` for initial
flux, and zero background.
"""
function _extract_source_catalog(mf::MatchedFilterResult, psf, ::Type{T}) where {T}
    n = length(mf.peaks)
    y  = T[Tuple(p)[1] for p in mf.peaks]
    x  = T[Tuple(p)[2] for p in mf.peaks]
    flux = T.(mf.peak_fluxes)
    bkg = zeros(T, n)
    return _build_params_matrix(psf, (; y, x, flux, bkg), T)
end

# ==============================================================================
# Error extraction
# ==============================================================================

"""
    _extract_errors!(errors, cov, free_idx, is_fixed, i::Int)

Extract 1-sigma parameter errors from the `(n_free × n_free)` covariance
matrix `cov` into column `i` of `errors`.  Free-parameter errors are
`sqrt(cov[j, j])`; fixed-parameter errors are set to `zero(T)`.
"""
function _extract_errors!(errors::Matrix{T}, cov, free_idx, is_fixed, i::Int) where {T}
    for (j, k) in enumerate(free_idx)
        errors[k, i] = T(sqrt(max(zero(T), cov[j, j])))
    end
    for k in is_fixed
        errors[k, i] = zero(T)
    end
    return errors
end

# ==============================================================================
# Core algorithm
# ==============================================================================

"""
    fit_all_stars(image, psf, sources; kws...) -> MultiPassPhotResult

Perform multi-pass PSF-fitting photometry on all sources in `image`.

Stars are sorted by brightness and fitted one at a time against a
progressive residual image.  Each fitted model is subtracted before the
next star is processed, so fainter neighbours are measured after brighter
stars have been removed.  On subsequent passes each star is added back,
re-fitted, and re-subtracted, progressively refining all measurements.

# Arguments

- `image::AbstractMatrix`: the background-subtracted image.
- `psf::AbstractPSFModel`: PSF model shared by all stars.  Its structural
  parameters (FWHM, shape, etc.) are fixed; `y`, `x`, `flux`, `bkg` are
  fitted per star (unless frozen via `fixed`).
- `sources`: source catalog.  Accepted formats:
  - `Vector{<:NamedTuple}` — output of [`measure_star_shapes`](@ref).
  - `NamedTuple` with `(:y, :x)` fields and optional `(:flux, :bkg)`.
  - [`MatchedFilterResult`](@ref).

# Keyword arguments

- `n_passes::Integer = 3`: number of full subtract/add-back/re-fit cycles.
- `inv_var`: inverse-variance map with the same shape as `image`.  Must be
  finite and positive on the fitting indices of each source.
- `fixed::NamedTuple = (;)`: parameters frozen for ALL stars, e.g.
  `(; bkg)` or `(; x, y)`.
- `fwhm_factor::Real = 5`: controls the pixel footprint around each star
  via `CartesianIndices(model, fwhm_factor)`.
- Remaining keywords (`max_iter`, `x_tol`, `f_tol`, `g_tol`, `show_trace`,
  `reweight`, `covariance_estimator`, `scale_estimator`, `damping`) are
  forwarded to [`fit_star`](@ref CrowdPhot.PSF.fit_star).

# Returns

A [`MultiPassPhotResult`](@ref) containing fitted parameters, errors,
convergence flags, and the final residual image.
"""
function fit_all_stars(
        image::AbstractMatrix{T},
        psf::AbstractPSFModel,
        sources;
        n_passes::Integer = 3,
        inv_var = nothing,
        fixed::NamedTuple = (;),
        fwhm_factor::Real = 5,
        kws...) where {T}
    #     max_iter::Integer = 200,
    #     x_tol::Real = 1.0e-8,
    #     f_tol::Real = 1.0e-8,
    #     g_tol::Real = 1.0e-8,
    #     show_trace::Bool = false,
    #     reweight = nothing,
    #     scale_estimator = nothing,
    #     covariance_estimator = nothing,
    #     damping::AbstractLMDamping = MarquardtDamping(),
    # ) where {T}
    FT = float(T)

    # -------------------------------------------------------------------
    # 1. Extract source catalog into contiguous params/errors matrices
    # -------------------------------------------------------------------
    params, errors = _extract_source_catalog(sources, psf, FT)
    n_params, n_stars = size(params)
    n_stars == 0 && return MultiPassPhotResult(
        FT[], FT[], FT[], FT[], FT[], FT[], FT[], FT[],
        BitVector[], BitVector[], FT[], Int(0), Matrix{FT}(undef, 0, 0),
    )

    # Map PSF property names to matrix row indices.
    prop_names = collect(keys(ConstructionBase.getproperties(psf)))
    @assert length(prop_names) == n_params

    # Identify which rows hold y, x, flux, bkg for sorting and validation.
    row_y    = findfirst(==(:y), prop_names)
    row_x    = findfirst(==(:x), prop_names)
    row_flux = findfirst(==(:flux), prop_names)
    row_bkg  = findfirst(==(:bkg), prop_names)

    # -------------------------------------------------------------------
    # 2. Free-parameter metadata (computed once)
    # -------------------------------------------------------------------
    free_names, free_idx, _ = PSF.free_params(psf, fixed)
    isempty(free_idx) && throw(ArgumentError("all model parameters are fixed; nothing to fit"))
    is_fixed = Tuple(setdiff(1:n_params, free_idx))

    # Set fixed-parameter errors to zero immediately (won't change).
    for k in is_fixed
        errors[k, :] .= zero(FT)
    end

    # -------------------------------------------------------------------
    # 3. Copy image for progressive subtraction
    # -------------------------------------------------------------------
    residual = Matrix{FT}(image)

    # -------------------------------------------------------------------
    # 4. Per-star state vectors
    # -------------------------------------------------------------------
    valid     = trues(n_stars)
    converged = falses(n_stars)
    chisq     = zeros(FT, n_stars)

    # -------------------------------------------------------------------
    # 5. Multi-pass loop
    # -------------------------------------------------------------------
    for pass in 1:n_passes
        # Sort by flux descending (brightest first) so brighter stars
        # are subtracted before fainter neighbours are fitted.
        order = sortperm(view(params, row_flux, :); rev = true)

        for idx in order
            valid[idx] || continue

            # Build model from all parameters in the params column.
            # Fixed parameters retain their per-star initial values.
            all_vals = NamedTuple{Tuple(prop_names)}(ntuple(k -> params[k, idx], Val(n_params)))
            m = ConstructionBase.setproperties(psf, all_vals)

            # Pixel footprint, clamped to image bounds.
            inds = _clamp_inds(CartesianIndices(m, fwhm_factor), residual)
            length(inds) < 3 && (valid[idx] = false; continue)

            # On passes 2+, add this star's previous model back so it is
            # fitted against data containing only its own signal (plus
            # noise); all neighbours are already subtracted.
            if pass > 1
                PSF.add_star!(residual, m, inds)
            end

            # Fit the star on the residual image.
            best, result = PSF.fit_star(
                m, residual, inds;
                kws...
                # fixed, inv_var,
                # max_iter, x_tol, f_tol, g_tol,
                # show_trace, reweight,
                # scale_estimator, covariance_estimator,
                # damping,
            )

            return best, result

            # Update all parameter rows from the fitted model.
            best_props = ConstructionBase.getproperties(best)
            for k in 1:n_params
                params[k, idx] = FT(getfield(best_props, prop_names[k]))
            end

            # Extract errors from the covariance matrix.
            if result.cov !== nothing
                _extract_errors!(errors, result.cov, free_idx, is_fixed, idx)
            end

            converged[idx] = result.converged
            chisq[idx]     = result.chisq

            # Subtract the updated best-fit model.
            PSF.subtract_star!(residual, best, inds)
        end

        # Between-pass validation: reject non-positive / NaN flux,
        # non-converged fits.
        for idx in 1:n_stars
            valid[idx] &= converged[idx] &&
                          isfinite(params[row_flux, idx]) &&
                          params[row_flux, idx] > 0
        end
    end

    # -------------------------------------------------------------------
    # 6. Assemble result
    # -------------------------------------------------------------------
    y  = params[row_y, :]
    x  = params[row_x, :]
    flux = params[row_flux, :]
    bkg  = params[row_bkg, :]

    y_err  = errors[row_y, :]
    x_err  = errors[row_x, :]
    flux_err = errors[row_flux, :]
    bkg_err  = errors[row_bkg, :]

    return MultiPassPhotResult(
        y, x, y_err, x_err, flux, flux_err, bkg, bkg_err,
        converged, valid, chisq, n_passes, residual,
    )
end
