using CrowdPhot
using CrowdPhot: CircularGaussianPSF, StampDerivatives, NeighborBlocks,
    _extract_source_catalog, _build_stamps!, _build_neighbors, _fill_stamps!,
    _accumulate_H!, _build_cholesky_cache, _refill_cholesky!,
    selected_inverse, selected_inverse_diagonal_blocks
using CrowdPhot.PSF
using ConstructionBase
using SparseArrays
using LinearAlgebra
using StableRNGs
using Test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Rebuild the exact full Gauss-Newton Hessian (diagonal blocks + off-diagonal
# neighbor-coupling blocks, in physical (y, x, flux) units) that
# `fit_all_stars_simultaneous` assembles internally at a fixed parameter
# point `fitted = (; y, x, flux)`, using the same private stamp/neighbor
# machinery it uses. Returns `(H, Hdiag, nb, p, active)`, with `H` a sparse
# matrix storing only its lower triangle (matching `CholeskySolverCache.H`'s
# convention, i.e. what `selected_inverse` expects).
function _rebuild_fit_hessian(image, psf, fitted, fit_rad, inv_var, fixed)
    FT = Float64
    params, _ = _extract_source_catalog(fitted, psf, FT)
    prop_names = collect(keys(ConstructionBase.getproperties(psf)))
    free_names, free_idx, _ = PSF.free_params(psf, fixed)
    p = length(free_idx)
    row_y = findfirst(==(:y), prop_names)
    row_x = findfirst(==(:x), prop_names)
    row_flux = findfirst(==(:flux), prop_names)
    grad_col = [free_names[k] === :y ? 1 : (free_names[k] === :x ? 2 : 3) for k in 1:p]
    free_names_val = Val(free_names)

    ny, nx = size(image)
    npix = ny * nx
    w = Vector{FT}(undef, npix)
    for idx in eachindex(image)
        wv = inv_var[idx]
        w[idx] = isfinite(wv) && wv > 0 ? FT(wv) : zero(FT)
    end

    built = _build_stamps!(image, params, row_y, row_x, row_flux, fit_rad, w, FT)
    pixels, anchor_y, anchor_x = built.pixels, built.anchor_y, built.anchor_x
    dy_off, dx_off, active, S2 = built.dy_off, built.dx_off, built.active, built.S2
    n_active = length(active)

    _, nb = _build_neighbors(pixels, S2, n_active, p, FT)

    θ = zeros(FT, p * n_active)
    for (j, i) in enumerate(active), k in 1:p
        θ[(j - 1) * p + k] = params[free_idx[k], i]
    end

    stamp = StampDerivatives{FT, Int32}(zeros(FT, p, S2, n_active), pixels, zeros(FT, p, n_active), npix, p, S2)
    model_img = zeros(FT, npix)
    Hdiag = zeros(FT, p, p, n_active)
    _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w, model_img, grad_col,
        dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, trues(n_active), nothing)
    _accumulate_H!(Hdiag, nb, stamp)

    # `_accumulate_H!` operates on `_fill_stamps!`'s column-equilibrated
    # Jacobian; undo the equilibration to recover physical-units Gauss-Newton
    # blocks, mirroring what `fit_all_stars_simultaneous`'s own final error
    # block does to `blk` before inverting it.
    for a in 1:n_active, l in 1:p, k in 1:p
        Hdiag[k, l, a] *= stamp.colnorm[k, a] * stamp.colnorm[l, a]
    end
    for t in 1:length(nb)
        a, b = nb.pair_a[t] + 1, nb.pair_b[t] + 1
        for l in 1:p, k in 1:p
            nb.B[k, l, t] *= stamp.colnorm[k, a] * stamp.colnorm[l, b]
        end
    end

    cache = _build_cholesky_cache(nb, n_active, p)
    _refill_cholesky!(cache, Hdiag, nb, trues(n_active))
    return cache.H, Hdiag, nb, p, active
end

# A minimal two-star, noisy, circular-Gaussian scene built directly from
# `PSF.evaluate`, independent of `simulate_image`'s random source placement,
# so the exact separation between the two stars can be controlled.
function _two_star_image(rng, ny, nx, y1, x1, f1, y2, x2, f2, fwhm, σ)
    FT = Float64
    img = zeros(FT, ny, nx)
    psf1 = CircularGaussianPSF(y1, x1, fwhm, f1, FT(0))
    psf2 = CircularGaussianPSF(y2, x2, fwhm, f2, FT(0))
    for xi in 1:nx, yi in 1:ny
        img[yi, xi] += PSF.evaluate(psf1, yi, xi) + PSF.evaluate(psf2, yi, xi)
    end
    img .+= σ .* randn(rng, ny, nx)
    return img
end

@testset "sparse selected inverse (Takahashi recursion)" begin

    @testset "matches dense inverse at the fill-in pattern" begin
        rng = StableRNG(1)
        n = 14
        A = zeros(n, n)
        for i in 1:n, j in 1:n
            abs(i - j) <= 3 && (A[i, j] = randn(rng))
        end
        A = (A + A') / 2 + n * I
        Hlow = sparse(LowerTriangular(A))
        Σ = selected_inverse(Hlow)
        Adense = inv(A)
        @test nnz(Σ) > 0
        for j in 1:n, i in 1:n
            if Σ[i, j] != 0
                @test Σ[i, j] ≈ Adense[i, j] rtol = 1e-8
            end
        end
        # Every diagonal entry of a Cholesky factor is structurally
        # nonzero, so every diagonal entry of A is always recovered.
        @test all(Σ[i, i] ≈ Adense[i, i] for i in 1:n)
    end

    @testset "block-diagonal H: selected inverse matches per-block inverse exactly" begin
        rng = StableRNG(2)
        p = 3
        nblocks = 5
        n = p * nblocks
        A = zeros(n, n)
        for b in 1:nblocks
            base = (b - 1) * p
            blk = randn(rng, p, p)
            blk = blk * blk' + p * I
            A[(base + 1):(base + p), (base + 1):(base + p)] .= blk
        end
        Hlow = sparse(LowerTriangular(A))
        blocks = selected_inverse_diagonal_blocks(Hlow, p)
        for b in 1:nblocks
            base = (b - 1) * p
            @test blocks[:, :, b] ≈ inv(A[(base + 1):(base + p), (base + 1):(base + p)]) rtol = 1e-10
        end
    end

    @testset "coupled 2-block H: marginal variance >= diagonal-block-only variance" begin
        # Schur-complement inequality: for H = [A C; C' B], the (1,1) block
        # of inv(H) is (A - C*inv(B)*C')^{-1} >= inv(A) in the Loewner order
        # whenever C != 0 -- ignoring coupling always understates the true
        # marginal variance, and the gap grows with the coupling strength.
        A11 = [3.0 0.2; 0.2 2.5]
        A22 = [2.8 -0.1; -0.1 3.1]
        for coupling in (0.0, 0.5, 1.5)
            C = coupling .* [0.5 0.1; 0.05 0.4]
            H = [A11 C; C' A22]
            blocks = selected_inverse_diagonal_blocks(sparse(LowerTriangular(H)), 2)
            @test blocks[:, :, 1] ≈ inv(H)[1:2, 1:2] rtol = 1e-10
            @test blocks[:, :, 2] ≈ inv(H)[3:4, 3:4] rtol = 1e-10
            for k in 1:2
                @test blocks[k, k, 1] >= inv(A11)[k, k] - 1e-12
                @test blocks[k, k, 2] >= inv(A22)[k, k] - 1e-12
            end
            if coupling > 0
                @test blocks[1, 1, 1] > inv(A11)[1, 1]
                @test blocks[1, 1, 2] > inv(A22)[1, 1]
            end
        end
    end

    @testset "uncrowded field: Takahashi errors approximately equal fit_all_stars_simultaneous" begin
        rng = StableRNG(7)
        ny = nx = 100
        fwhm, σ, fit_rad = 3.0, 5.0, 5
        y1, x1, f1 = 50.0, 40.0, 8000.0
        y2, x2, f2 = 50.0, 80.0, 8000.0 # separation 40px >> 2*fit_rad+1: stamps never share a pixel
        img = _two_star_image(rng, ny, nx, y1, x1, f1, y2, x2, f2, fwhm, σ)
        inv_var = fill(1 / σ^2, ny, nx)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=fwhm, flux=1.0, bkg=0.0)
        fixed = (; fwhm=fwhm, bkg=0.0)
        cat = (; y=[y1, y2], x=[x1, x2], flux=[f1, f2])

        r = fit_all_stars_simultaneous(img, psf, cat, fit_rad;
            fixed, inv_var, solver=:cholesky, max_iter=40, x_tol=1e-9)
        @test all(r.valid)

        fitted = (; y=r.y, x=r.x, flux=r.flux)
        H, Hdiag, nb, p, active = _rebuild_fit_hessian(img, psf, fitted, fit_rad, inv_var, fixed)
        @test length(nb) == 0 # no shared pixels: H is exactly block-diagonal

        blocks = selected_inverse_diagonal_blocks(H, p)
        for (j, i) in enumerate(active)
            diag_only = inv(Hdiag[:, :, j])
            marginal = blocks[:, :, j]
            # Exactly equal in exact arithmetic whenever H is block-diagonal:
            # the inverse of a block-diagonal matrix is block-diagonal, with
            # each block the inverse of the corresponding diagonal block.
            @test marginal ≈ diag_only rtol = 1e-8
            # Both match what fit_all_stars_simultaneous itself reports, up
            # to its 1e-12*trace conditioning term (a part-in-1e4 effect on
            # this fixture's well-conditioned flux block).
            @test marginal[3, 3] ≈ r.flux_err[i]^2 rtol = 1e-3
            @test marginal[1, 1] ≈ r.y_err[i]^2 rtol = 1e-3
            @test marginal[2, 2] ≈ r.x_err[i]^2 rtol = 1e-3
        end
    end

    @testset "crowded field: Takahashi marginal errors exceed fit_all_stars_simultaneous's diagonal-only errors" begin
        rng = StableRNG(11)
        ny = nx = 100
        fwhm, σ, fit_rad = 3.0, 5.0, 6
        y1, x1, f1 = 50.0, 47.0, 8000.0
        y2, x2, f2 = 50.0, 50.0, 8000.0 # separation 3px: well within the fit box and the PSF wings
        img = _two_star_image(rng, ny, nx, y1, x1, f1, y2, x2, f2, fwhm, σ)
        inv_var = fill(1 / σ^2, ny, nx)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=fwhm, flux=1.0, bkg=0.0)
        fixed = (; fwhm=fwhm, bkg=0.0)
        cat = (; y=[y1, y2], x=[x1, x2], flux=[f1, f2])

        r = fit_all_stars_simultaneous(img, psf, cat, fit_rad;
            fixed, inv_var, solver=:cholesky, max_iter=40, x_tol=1e-9)
        @test all(r.valid)

        fitted = (; y=r.y, x=r.x, flux=r.flux)
        H, Hdiag, nb, p, active = _rebuild_fit_hessian(img, psf, fitted, fit_rad, inv_var, fixed)
        @test length(nb) > 0 # the two stamps do share pixels here

        blocks = selected_inverse_diagonal_blocks(H, p)
        # Cross-check against a dense inverse of the same (small, 6x6) H: the
        # selected-inversion result must be the exact marginal, not merely
        # "larger than the diagonal-only estimate".
        Σdense = inv(Matrix(Symmetric(H, :L)))
        for (j, i) in enumerate(active)
            base = (j - 1) * p
            @test blocks[:, :, j] ≈ Σdense[(base + 1):(base + p), (base + 1):(base + p)] rtol = 1e-8

            diag_only_flux_var = inv(Hdiag[:, :, j])[3, 3]
            marginal_flux_var = blocks[3, 3, j]
            @test marginal_flux_var > diag_only_flux_var
            # The docstring this PR is validating flags the diagonal-only
            # figure as an underestimate; for this strongly-blended pair the
            # true marginal must exceed it by a non-trivial amount, not just
            # numerical noise.
            @test marginal_flux_var > 1.1 * r.flux_err[i]^2
        end
    end
end
