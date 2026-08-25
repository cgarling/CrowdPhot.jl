# Shared per-star goodness-of-fit diagnostics for PSF photometry.
#
# `fit_all_stars` (sequential) and `fit_all_stars_simultaneous` compute the
# same qfit / qfit_expected / qfit_z / crowding statistics so their results
# compare field-for-field.  The math below is extracted verbatim from the
# sequential path (psf_photometry_single.jl) so both arms share one
# implementation; only the "neighbor-subtracted residual" input differs.

"""
    _star_diagnostics!(qfit, qfit_expected, qfit_z, crowding, idx,
                       model, image, clean_resid, inds, inv_var, n_free)

Compute the per-star `qfit`, `qfit_expected`, `qfit_z`, and `crowding`
diagnostics for star `idx` and write them into the corresponding vectors.

`clean_resid` must be the *neighbor-subtracted* residual image: the observed
data with every other star's best-fit model removed, but with this star's own
model still present.  For the sequential path this is the working residual at
diagnostics time; for the simultaneous path it is `global_residual + own_model`.

`image` is the original data (used for the `crowding` "dirty" flux), `model`
is the star's best-fit model, `inds` is the fitting box, and `n_free` is the
number of free parameters (used to correct `qfit_z` for fitting leverage).
"""
function _star_diagnostics!(
        qfit::AbstractVector, qfit_expected::AbstractVector, qfit_z::AbstractVector,
        crowding::AbstractVector, idx::Int, model, image, clean_resid,
        inds, inv_var, n_free::Int
    )
    FT = float(eltype(image))
    flux = FT(model.flux)
    bkg = FT(model.bkg)
    flux > 0 || return nothing

    inv_flux = inv(flux)
    qfit_val = zero(FT)
    num_clean = zero(FT)
    num_dirty = zero(FT)
    den_crowd = zero(FT)
    for pix in inds
        model_val = evaluate(model, pix)
        wp = inv_var !== nothing ? inv_var[pix] : one(FT)
        if isfinite(wp) && wp > 0
            qfit_val += abs(clean_resid[pix] - model_val)
            # Unit-flux PSF kernel for the crowding calculation.
            Pp = (model_val - bkg) * inv_flux
            wP = wp * Pp
            num_clean += wP * (clean_resid[pix] - bkg)
            num_dirty += wP * (image[pix] - bkg)
            den_crowd += wP * Pp
        end
    end
    qfit[idx] = qfit_val * inv_flux
    if den_crowd > 0 && num_clean > 0 && num_dirty > 0
        crowding[idx] = FT(2.5) * log10(num_dirty / num_clean)
    end

    # qfit_expected and qfit_z depend only on inv_var and the box.
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
        if sigma2_sum > 0 && n_pix_good > n_free
            dof_factor = FT(sqrt(1 - n_free / n_pix_good))
            num = qfit_val - FT(sqrt(2 / FT(π))) * dof_factor * sigma_sum
            den = FT(sqrt((1 - 2 / FT(π)) * sigma2_sum)) * dof_factor
            qfit_z[idx] = num / den
        end
    end
    return nothing
end
