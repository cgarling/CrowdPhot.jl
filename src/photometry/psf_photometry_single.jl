# PSF-fitting photometry for a single image.
#
# Implements a DOLPHOT-style multi-pass algorithm: stars are sorted by
# brightness and fitted one at a time against a residual image from which
# all brighter (and, on later passes, all other) stars have been subtracted.
# On passes 2+, each star is added back before re-fitting so it sees the
# original data minus only its neighbors.

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
- `finalized::BitVector`: whether the source was selected for the "finalize"
  step (see `finalize_snr_min`/`finalize_rad` keyword arguments below).
  Decided once, from the pass-1 fit, and never revisited. For finalized
  stars, `flux`/`flux_err` come from the closed-form large-footprint
  estimator rather than the small-footprint LM fit.
- `n_iter::Vector{Int}`: total LM iterations accumulated across all passes for each source.
- `n_passes::Int`: number of passes actually performed.
- `n_failed::Int`: number of stars whose fits threw exceptions and were marked
  invalid.  A non-zero count indicates systematic problems (e.g. shape
  mismatches in `inv_var`) and a warning is emitted.
- `failure_msgs::Vector{String}`: first few exception messages from failed fits,
  for diagnosis.  Empty when `n_failed == 0`.
- `residual::Matrix{T}`: final residual image after all subtractions.

# Goodness-of-fit diagnostics
- `chisq::Vector{T}`: final reduced χ² for each source.  Uses squared residuals (L2 norm), so a
  single bad pixel (cosmic ray, hot pixel) can dominate.  Best for catching
  single-pixel excursions.
- `qfit::Vector{T}`: sum of absolute residuals over valid-weight pixels in the
  fitting aperture divided by flux.  Uses L1 norm, so it is robust to single-pixel
  outliers and sensitive to persistent misfit across many pixels (PSF mismatch,
  extended sources).  Computed on the final pass from the best-fit model;
  check `valid` and `converged` before interpreting.  Same statistic as in
  ``\\texttt{hst1pass}``, see [Anderson2022hst1pass](@citet) page 10.
- `qfit_expected::Vector{T}`: expected `qfit` under the noise model
  ``\\left(\\sqrt{2/\\pi} \\sum \\sigma_i \\,/\\, \\mathrm{F}\\right)``.
  The difference and/or ratio between `qfit` and `qfit_expected` can be used to
  identify stars whose fits are worse than the noise model predicts, though `qfit_z`
  is likely to be more informative. `NaN` when `inv_var` was not provided.
- `qfit_z::Vector{T}`: noise-standardized excess absolute residual:
  ```math
  q_{\\rm fit,z} =
  \\frac{\\sum |r_i| - \\sqrt{\\frac{2}{\\pi}(1 - p/N)} \\sum \\sigma_i}
  {\\sqrt{(1 - \\frac{2}{\\pi})(1 - p/N) \\sum \\sigma_i^2}},
  \\qquad
  r_i = P_i - s - F\\psi_i .
  ```
  where ``p`` is the number of free parameters and ``N`` is the number of
  valid-weight pixels in the fitting aperture.  The ``\\sqrt{1 - p/N}``
  factors correct for the reduction in residual variance due to fitting:
  with ``p`` parameters estimated from ``N`` data points, each residual's
  variance is reduced by approximately ``(1 - p/N)`` (average leverage
  ``h_{ii} \\approx p/N``).  The expectation of ``|r_i|`` for a half-normal
  scales with the standard deviation, so both the numerator expectation and
  the denominator standard deviation are multiplied by ``\\sqrt{1 - p/N}``.
  Under the null hypothesis, ``q_{\\rm fit,z} \\sim \\mathcal{N}(0,1)`` to
  within ``\\mathcal{O}(p^2/N^2)``.  Large positive values indicate
  structured residuals inconsistent with pure noise.  `NaN` when `inv_var`
  was not provided or when the aperture has fewer valid-weight pixels than
  free parameters.
- `crowding::Vector{T}`: DOLPHOT-like blend-contamination diagnostic:
  ``2.5 \\log_{10}(F_{\\rm dirty} / F_{\\rm clean})`` where ``F_{\\rm clean}``
  is measured on the neighbor-subtracted image and ``F_{\\rm dirty}`` is
  measured on the original image with all neighbors present.  Values near zero
  indicate an isolated source; positive values indicate contamination by
  neighbor light.  Computed on the final pass; check `valid` and `converged`
  before interpreting.  For finalized stars (see `finalized` above), this is
  computed over the large finalize footprint (the same region that produced
  `flux`); otherwise it uses the small fitting footprint.
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
    finalized::BitVector
    chisq::Vector{T}
    qfit::Vector{T}
    qfit_expected::Vector{T}
    qfit_z::Vector{T}
    crowding::Vector{T}
    n_iter::Vector{Int}
    n_passes::Int
    n_failed::Int
    failure_msgs::Vector{String}
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
`matched_filter_flux` as the initial flux guess; `bkg` defaults to zero.
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
Uses `Tuple.(peaks)` for pixel-center `(y, x)`, `peak_fluxes` for initial
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

# Footprint (as `CartesianIndices`, unclamped) used for the "finalize" step
# for a given model instance. If `finalize_rad` is `nothing`, use the
# model's own natural extent (the full tabulated grid for
# `ImagePSF`/`GriddedPSFModel`, or a `5xFWHM` box for analytic models);
# otherwise a `±finalize_rad` square box centered on the model's position.
function _finalize_extent(model, finalize_rad::Union{Nothing, Real})
    if isnothing(finalize_rad)
        return CartesianIndices(model)
    else
        yr = floor(Int, model.y - finalize_rad):ceil(Int, model.y + finalize_rad)
        xr = floor(Int, model.x - finalize_rad):ceil(Int, model.x + finalize_rad)
        return CartesianIndices((yr, xr))
    end
end

# Closed-form weighted-least-squares flux sums for `model` (with `y`, `x`,
# `bkg`, and shape held fixed) over `inds`, evaluated against both `clean`
# (e.g. the neighbor-subtracted residual) and `dirty` (e.g. the original
# image with all neighbors present) data. Returns `(num_clean, num_dirty,
# den)` where `num_clean / den` is the flux MLE and `1 / den` is its
# variance; `num_dirty / den` is the corresponding "dirty" flux, used for
# the `crowding` diagnostic. This is the same accumulation used for the
# existing `crowding` computation, generalized to an arbitrary footprint.
function _finalize_sums(
        model, clean::AbstractMatrix{FT}, dirty::AbstractMatrix, inds, inv_var,
    ) where {FT}
    inv_flux = inv(model.flux)
    bkg = model.bkg
    num_clean = zero(FT)
    num_dirty = zero(FT)
    den = zero(FT)
    for pix in inds
        model_val = evaluate(model, pix)
        w = inv_var !== nothing ? inv_var[pix] : one(FT)
        if isfinite(w) && w > 0
            Pp = (model_val - bkg) * inv_flux
            wP = w * Pp
            num_clean += wP * (clean[pix] - bkg)
            num_dirty += wP * (dirty[pix] - bkg)
            den += wP * Pp
        end
    end
    return num_clean, num_dirty, den
end

"""
    fit_all_stars(image, psf, sources, fit_rad; kws...) -> MultiPassPhotResult

Perform multi-pass PSF-fitting photometry on all sources in `image`.

Stars are sorted by brightness and fitted one at a time against a
progressive residual image.  Each fitted model is subtracted before the
next star is processed, so fainter neighbors are measured after brighter
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
- `fit_rad::Real`: fitting radius in detector pixels.  A ±`fit_rad`
  rectangular cutout is extracted around each star's current position.

# Keyword arguments

- `n_passes::Integer = 3`: number of full subtract/add-back/re-fit cycles.
- `fixed::NamedTuple = (;)`: parameters frozen for ALL stars, e.g.
  `(; bkg)` or `(; x, y)`.
- `inv_var`: per-pixel inverse variance for weighted fitting.  Must be the same
  shape as `image`.  Forwarded to [`fit_star`](@ref CrowdPhot.PSF.fit_star),
  where non-positive or non-finite values are treated as masked pixels.  Stars
  with too few valid pixels are marked invalid and skipped.
- `finalize_snr_min::Real = Inf`: SNR threshold (flux / flux_err from the
  small-footprint fit) above which a star is selected for the "finalize"
  step. The decision is made **once**, from the pass-1 fit, and never
  revisited on later passes. Disabled by default (`Inf`).  For a selected
  star, on *every* pass (not just the last), after the ordinary small-
  footprint fit: the star's flux and flux uncertainty are re-measured with a
  closed-form weighted-least-squares sum (no LM iteration; position,
  background, and shape are held fixed at their small-footprint fitted
  values) evaluated over a larger footprint (see `finalize_rad`), and that
  larger footprint — not the small fitting footprint — is what gets
  subtracted from the residual. This both (a) uses more of the PSF model's
  information content to shrink `flux_err` for bright stars, and (b) removes
  more of a bright star's wings from the residual before fainter neighbors
  in its vicinity are measured, on every subsequent pass. `qfit`,
  `qfit_expected`, and `qfit_z` remain scoped to the small fitting footprint
  regardless of finalize status (a large-footprint aggregate would be
  dominated by wing noise and lose sensitivity to core defects); `crowding`
  tracks whichever footprint produced the reported flux.  Requires `flux` to
  be a free parameter (not in `fixed`).
- `finalize_rad::Union{Nothing,Real} = nothing`: half-width, in detector
  pixels, of the `±finalize_rad` square footprint used for the finalize
  step. Must satisfy `finalize_rad >= fit_rad` when given. If `nothing`
  (the default), the model's own natural extent (`CartesianIndices(psf)`) is
  used instead — the full tabulated grid for `ImagePSF`/`GriddedPSFModel`,
  or a `5×FWHM` box for analytic models (with a warning, since an analytic
  model's infinite tails make its default extent a somewhat arbitrary choice
  of "large" footprint — an explicit `finalize_rad` is recommended for such
  models). Ignored when `finalize_snr_min = Inf`.
- All other keyword arguments (`max_iter`, `x_tol`, `f_tol`,
  `g_tol`, `show_trace`, `reweight`, `covariance_estimator`,
  `scale_estimator`, `damping`, etc.) are forwarded to
  [`fit_star`](@ref CrowdPhot.PSF.fit_star).

# Returns

A [`MultiPassPhotResult`](@ref) containing fitted parameters, errors,
convergence flags, and the final residual image.
"""
function fit_all_stars(
        image::AbstractMatrix{T},
        psf::AbstractPSFModel,
        sources,
        fit_rad::Real;
        n_passes::Integer = 3,
        fixed::NamedTuple = (;),
        inv_var = nothing,
        finalize_snr_min::Real = Inf,
        finalize_rad::Union{Nothing, Real} = nothing,
        kws...,
    ) where {T}
    FT = float(T)
    n_passes > 0 || throw(ArgumentError("n_passes must be positive, got $n_passes"))
    isnothing(finalize_rad) || finalize_rad >= fit_rad ||
        throw(ArgumentError("finalize_rad ($finalize_rad) must be >= fit_rad ($fit_rad)"))
    finalize_rad_FT = isnothing(finalize_rad) ? nothing : FT(finalize_rad)

    # -------------------------------------------------------------------
    # 1. Extract source catalog into contiguous params/errors matrices
    # -------------------------------------------------------------------
    params, errors = _extract_source_catalog(sources, psf, FT)
    n_params, n_stars = size(params)
    n_stars == 0 && return MultiPassPhotResult(
        FT[], FT[], FT[], FT[], FT[], FT[], FT[], FT[],
        falses(0), falses(0), falses(0), FT[], FT[], FT[], FT[], FT[], Int[], Int(0), Int(0), String[], Matrix{FT}(undef, 0, 0),
    )

    # Map PSF property names to matrix row indices.
    prop_names = collect(keys(ConstructionBase.getproperties(psf)))
    @assert length(prop_names) == n_params

    # Identify which rows hold y, x, flux, bkg for sorting and validation.
    row_y = findfirst(==(:y), prop_names)
    row_x = findfirst(==(:x), prop_names)
    row_flux = findfirst(==(:flux), prop_names)
    row_bkg = findfirst(==(:bkg), prop_names)

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
    # 2b. Finalize-step setup (validated/precomputed once)
    # -------------------------------------------------------------------
    finalize_enabled = isfinite(finalize_snr_min)
    flux_free_pos = findfirst(==(row_flux), free_idx)
    if finalize_enabled && isnothing(flux_free_pos)
        @warn "finalize_snr_min is finite but `flux` is fixed (via `fixed`); " *
              "disabling the finalize step since an SNR cannot be computed for a fixed flux."
        finalize_enabled = false
    end
    if finalize_enabled && isnothing(finalize_rad_FT) && !PSF._has_finite_support(psf)
        @warn "finalize_rad is `nothing` and `psf` does not have finite support " *
              "(e.g. an analytic model with formally infinite tails); falling back to " *
              "the model's default `extent` (a 5xFWHM box) as the finalize footprint. " *
              "Pass an explicit `finalize_rad` to control this."
    end

    # -------------------------------------------------------------------
    # 3. Copy image for progressive subtraction
    # -------------------------------------------------------------------
    residual = Matrix{FT}(image)

    # -------------------------------------------------------------------
    # 4. Per-star state vectors
    # -------------------------------------------------------------------
    valid = trues(n_stars)
    converged = falses(n_stars)
    finalized = falses(n_stars)
    chisq = zeros(FT, n_stars)
    qfit = fill(convert(FT, NaN), n_stars)
    qfit_expected = fill(convert(FT, NaN), n_stars)
    qfit_z = fill(convert(FT, NaN), n_stars)
    crowding = fill(convert(FT, NaN), n_stars)
    n_iter = zeros(Int, n_stars)
    n_failed = 0
    failure_msgs = String[]

    # -------------------------------------------------------------------
    # 5. Multi-pass loop
    # -------------------------------------------------------------------
    for pass in 1:n_passes
        # Sort by flux descending (brightest first) so brighter stars
        # are subtracted before fainter neighbors are fitted.
        order = sortperm(view(params, row_flux, :); rev = true)

        for idx in order
            valid[idx] || continue

            # Build model from all parameters in the params column.
            # Fixed parameters retain their per-star initial values.
            all_vals = NamedTuple{Tuple(prop_names)}(ntuple(k -> params[k, idx], Val(n_params)))
            m = ConstructionBase.setproperties(psf, all_vals)

            # Pixel footprint of ±fit_rad around the star center, clamped
            # to image bounds.
            FT_fit = FT(fit_rad)
            yr = floor(Int, m.y - FT_fit):ceil(Int, m.y + FT_fit)
            xr = floor(Int, m.x - FT_fit):ceil(Int, m.x + FT_fit)
            inds = _clamp_inds(CartesianIndices((yr, xr)), residual)
            length(inds) < 3 && (valid[idx] = false; continue)

            # Fit the star on the residual image.  A failed fit (e.g. too
            # few valid inv_var pixels) marks the star invalid and continues
            # to the next source without crashing the full run.
            # `prev_inds` records the footprint that was actually subtracted
            # for this star the last time it was processed (small `inds` if
            # it was never finalized, or the finalize footprint if it was —
            # see `finalized` below); it is what must be added back on
            # passes 2+, and also what must be undone if this pass's fit
            # throws.
            prev_inds = (pass > 1 && finalized[idx]) ?
                        _clamp_inds(_finalize_extent(m, finalize_rad_FT), residual) : inds
            local best, result
            try
                # On passes 2+, add this star's previous model back so it is
                # fitted against data containing only its own signal (plus
                # noise); all neighbors are already subtracted.
                if pass > 1
                    PSF.add_star!(residual, m, prev_inds)
                end

                best, result = PSF.fit_star(
                    m, residual, inds;
                    fixed, inv_var, kws...,
                )

                # Update all parameter rows from the fitted model.
                best_props = ConstructionBase.getproperties(best)
                for k in 1:n_params
                    params[k, idx] = FT(getfield(best_props, prop_names[k]))
                end

                # Extract errors from the covariance matrix.
                if ~isnothing(result.cov)
                    _extract_errors!(errors, result.cov, free_idx, is_fixed, idx)
                end

                converged[idx] = result.converged
                chisq[idx] = result.chisq
                n_iter[idx] += result.iterations

                # Decide finalize-group membership exactly once, from the
                # pass-1 fit, and never revisit it. Pass-1 contamination
                # from not-yet-subtracted fainter neighbors (brighter stars
                # are already subtracted; fainter ones are not, this early
                # in the loop) can only inflate a star's apparent SNR, never
                # deflate it — so this is a safe, one-sided (inclusive, not
                # exclusive) selection. See `photometry_finalize.md` §4.3
                # for the full argument.
                # NOTE: selecting on pass 2 instead (after one round of
                # subtraction has reduced blend contamination in the SNR
                # estimate) might be preferable in the future; deferred here
                # to keep the selection logic simple (decide once, never
                # revisit).
                if pass == 1
                    snr = if finalize_enabled && !isnothing(result.cov)
                        var_flux = result.cov[flux_free_pos, flux_free_pos]
                        var_flux > 0 ? best.flux / sqrt(var_flux) : convert(FT, NaN)
                    else
                        convert(FT, NaN)
                    end
                    finalized[idx] = isfinite(snr) && snr >= FT(finalize_snr_min)
                end
                do_finalize = finalized[idx]

                # For finalized stars, re-measure flux/flux_err with a
                # closed-form sum over a larger footprint (position,
                # background, and shape fixed at their small-footprint
                # fitted values — no LM iteration). This is what actually
                # gets subtracted from the residual for such stars.
                local best_f, inds_f, num_clean_f, num_dirty_f, den_f
                if do_finalize
                    inds_f = _clamp_inds(_finalize_extent(best, finalize_rad_FT), residual)
                    num_clean_f, num_dirty_f, den_f = _finalize_sums(best, residual, image, inds_f, inv_var)
                    flux_f = den_f > 0 ? num_clean_f / den_f : best.flux
                    flux_err_f = den_f > 0 ? FT(sqrt(inv(den_f))) : convert(FT, NaN)
                    best_f = ConstructionBase.setproperties(best, (; flux = flux_f))
                    params[row_flux, idx] = flux_f
                    errors[row_flux, idx] = flux_err_f
                end

                # Compute qfit and crowding in one pixel pass on the final
                # pass, before subtraction.  The unit-flux PSF kernel is
                # extracted from the already-evaluated model value.
                #
                # `qfit`/`qfit_expected`/`qfit_z` always stay scoped to the
                # small `inds` footprint, regardless of finalize status: an
                # aggregate computed over a much larger footprint would be
                # dominated by wing-noise pixels that carry no information
                # about core model mismatch, diluting (or for `qfit_z`,
                # losing statistical power to detect) exactly the defects
                # these diagnostics exist to catch. `crowding`, by contrast,
                # tracks whichever footprint actually produced the reported
                # flux, since it is a statement about that specific
                # measurement rather than a general fit-quality diagnostic.
                if pass == n_passes && best.flux > 0
                    inv_flux = inv(best.flux)
                    qfit_val = zero(FT)
                    num_clean = zero(FT)
                    num_dirty = zero(FT)
                    den_crowd = zero(FT)
                    for pix in inds
                        model_val = evaluate(best, pix)
                        wp = inv_var !== nothing ? inv_var[pix] : one(FT)
                        if isfinite(wp) && wp > 0
                            qfit_val += abs(residual[pix] - model_val)
                            # Unit-flux PSF kernel for crowding calculation.
                            Pp = (model_val - best.bkg) * inv_flux
                            wP = wp * Pp
                            num_clean += wP * (residual[pix] - best.bkg)
                            num_dirty += wP * (image[pix] - best.bkg)
                            den_crowd += wP * Pp
                        end
                    end
                    qfit[idx] = qfit_val * inv_flux
                    if do_finalize
                        if den_f > 0 && num_clean_f > 0 && num_dirty_f > 0
                            crowding[idx] = FT(2.5) * log10(num_dirty_f / num_clean_f)
                        end
                    else
                        if den_crowd > 0 && num_clean > 0 && num_dirty > 0
                            crowding[idx] = FT(2.5) * log10(num_dirty / num_clean)
                        end
                    end
                    # qfit_expected and qfit_z (no evaluate, separate loop).
                    if !isnothing(inv_var)
                        sigma_sum = zero(FT)
                        sigma2_sum = zero(FT)
                        n_pix_good = 0
                        for pix in inds
                            iv = inv_var[pix]
                            if isfinite(iv) && iv > 0
                                sigma_i = inv(sqrt(iv))
                                sigma_sum += sigma_i
                                sigma2_sum += sigma_i^2
                                n_pix_good += 1
                            end
                        end
                        qfit_expected[idx] = FT(sqrt(2 / FT(π))) * sigma_sum * inv_flux
                        if sigma2_sum > 0 && n_pix_good > length(free_idx)
                            dof_factor = FT(sqrt(1 - length(free_idx) / n_pix_good))
                            num = qfit_val - FT(sqrt(2 / FT(π))) * dof_factor * sigma_sum
                            den = FT(sqrt((1 - 2 / FT(π)) * sigma2_sum)) * dof_factor
                            qfit_z[idx] = num / den
                        end
                    end
                end

                # Subtract the updated best-fit model over whichever
                # footprint applies (large finalize footprint if selected,
                # small fitting footprint otherwise).
                if do_finalize
                    PSF.subtract_star!(residual, best_f, inds_f)
                else
                    PSF.subtract_star!(residual, best, inds)
                end
            catch e
                # Undo the add-back so the residual stays consistent.
                if pass > 1
                    PSF.subtract_star!(residual, m, prev_inds)
                end
                valid[idx] = false
                n_failed += 1
                if length(failure_msgs) < 5
                    push!(failure_msgs, "star $idx (pass $pass): " * sprint(showerror, e))
                end
                continue
            end
        end

        # Between-pass validation: reject non-finite positions,
        # non-positive / NaN flux, non-converged fits.
        for idx in 1:n_stars
            valid[idx] &= converged[idx] &&
                          isfinite(params[row_y, idx]) &&
                          isfinite(params[row_x, idx]) &&
                          isfinite(params[row_flux, idx]) &&
                          params[row_flux, idx] > 0
        end
    end

    if n_failed > 0
        msg = "$n_failed / $n_stars star fits threw exceptions and were marked invalid"
        if !isempty(failure_msgs)
            msg *= ". Examples: " * join(failure_msgs, "; ")
        end
        @warn msg
    end

    # -------------------------------------------------------------------
    # 6. Assemble result
    # -------------------------------------------------------------------
    y = params[row_y, :]
    x = params[row_x, :]
    flux = params[row_flux, :]
    bkg = params[row_bkg, :]

    y_err = errors[row_y, :]
    x_err = errors[row_x, :]
    flux_err = errors[row_flux, :]
    bkg_err = errors[row_bkg, :]

    return MultiPassPhotResult(
        y, x, y_err, x_err, flux, flux_err, bkg, bkg_err,
        converged, valid, finalized, chisq, qfit, qfit_expected, qfit_z, crowding, n_iter, Int(n_passes), n_failed, failure_msgs, residual,
    )
end
