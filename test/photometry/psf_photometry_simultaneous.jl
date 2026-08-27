using CrowdPhot
using CrowdPhot: CircularGaussianPSF, StampDerivatives, apply_JT!, apply_J!,
    _accumulate_H!, _build_neighbors, _pcg!, _build_precond!, _apply_H!,
    NeighborBlocks, _clamp_inds,
    _build_cholesky_cache, _refill_cholesky!, _solve_cholesky!,
    _compact_neighbors, _mask_frozen!, _fill_stamps!
using CrowdPhot.PSF
using ConstructionBase
using LinearAlgebra: dot, norm, I, Symmetric
using StableRNGs
using Statistics: median, quantile
using Test

# Build the full (npix x n) Jacobian from a StampDerivatives so the adjoint
# identity and H accumulation can be checked against dense linear algebra.
function dense_J(stamp::StampDerivatives)
    p, S2, n_active = size(stamp.values)
    J = zeros(stamp.npix, n_active * p)
    for a in 0:(n_active - 1)
        for m in 1:S2
            fi = stamp.pixels[m, a + 1]
            fi != 0 || continue
            for k in 1:p
                J[fi, a * p + k] = stamp.values[k, m, a + 1]
            end
        end
    end
    return J
end

function assemble_H_dense(Hdiag, nb::NeighborBlocks, p, n_active)
    n = p * n_active
    H = zeros(n, n)
    for a in 0:(n_active - 1)
        for k in 1:p, l in 1:p
            H[a * p + k, a * p + l] = Hdiag[k, l, a + 1]
        end
    end
    for t in 1:length(nb)
        a = nb.pair_a[t]
        b = nb.pair_b[t]
        for k in 1:p, l in 1:p
            H[a * p + k, b * p + l] = nb.B[k, l, t]
            H[b * p + l, a * p + k] = nb.B[k, l, t]
        end
    end
    return H
end

# Column-equilibrate a stamp (as the real fill does) so H has unit diagonal.
function equilibrate!(stamp::StampDerivatives)
    p, S2, n_active = size(stamp.values)
    for a in 1:n_active
        for k in 1:p
            s = zero(eltype(stamp.values))
            for m in 1:S2
                s += stamp.values[k, m, a]^2
            end
            s = max(sqrt(s), eps())
            for m in 1:S2
                stamp.values[k, m, a] /= s
            end
        end
    end
    return stamp
end

@testset "simultaneous fitting" begin
    rng = StableRNG(1234)
    T = Float64

    @testset "_fill_stamps! preserves frozen colnorm/values" begin
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        fixed = (; fwhm=2.0, bkg=0.0)
        free_names, free_idx, _ = PSF.free_params(psf, fixed)
        p = length(free_idx)
        prop_names = collect(keys(ConstructionBase.getproperties(psf)))
        row_y = findfirst(==(:y), prop_names)
        row_x = findfirst(==(:x), prop_names)
        row_flux = findfirst(==(:flux), prop_names)
        grad_col = [free_names[k] === :y ? 1 : (free_names[k] === :x ? 2 : 3) for k in 1:p]
        free_names_val = Val(free_names)

        R = 1
        dy_off = Int[]
        dx_off = Int[]
        for dx in -R:R, dy in -R:R
            push!(dy_off, dy)
            push!(dx_off, dx)
        end
        S2 = length(dy_off)
        n_active = 2
        ny = 30
        anchor_y = [10, 10]
        anchor_x = [10, 20]  # far apart: no shared pixels between the two stars
        pixels = zeros(Int32, S2, n_active)
        for a in 1:n_active, m in 1:S2
            gy = anchor_y[a] + dy_off[m]
            gx = anchor_x[a] + dx_off[m]
            pixels[m, a] = gy + (gx - 1) * ny
        end
        npix = ny * 30
        θ = zeros(p * n_active)
        for a in 1:n_active
            θ[(a - 1) * p .+ (1:p)] .= (Float64(anchor_y[a]), Float64(anchor_x[a]), 100.0)
        end
        w = ones(npix)
        model_img = zeros(npix)
        stamp = StampDerivatives{Float64, Int32}(
            zeros(p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)

        live = trues(n_active)
        live[2] = false  # star 2 frozen from the start
        _fill_stamps!(stamp, psf, free_names_val, fixed, θ, w, model_img,
            grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, nothing)
        values_before = copy(stamp.values)
        colnorm_before = copy(stamp.colnorm)
        @test all(isfinite, values_before)
        @test all(isfinite, colnorm_before)

        # Repeated fills with a perturbed θ for the still-live star only: the
        # frozen star's column must stay bitwise untouched, not drift or blow
        # up via a stale colnorm floor (the bug this test regresses).
        θ2 = copy(θ)
        for _ in 1:5
            θ2[1] += 0.1
            _fill_stamps!(stamp, psf, free_names_val, fixed, θ2, w, model_img,
                grad_col, dy_off, dx_off, anchor_y, anchor_x, row_y, row_x, row_flux, live, nothing)
        end
        @test stamp.values[:, :, 2] == values_before[:, :, 2]
        @test stamp.colnorm[:, 2] == colnorm_before[:, 2]
        @test all(isfinite, stamp.values)
        @test all(isfinite, stamp.colnorm)
    end

    @testset "adjoint identity" begin
        p = 3
        S2 = 9
        n_active = 4
        npix = 20
        pixels = zeros(Int32, S2, n_active)
        # Give stars overlapping footprints.
        for a in 1:n_active
            for m in 1:S2
                fi = mod(m + (a - 1) * 2 - 1, npix) + 1
                pixels[m, a] = fi
            end
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        u = randn(rng, npix)
        v = randn(rng, p * n_active)
        y = zeros(npix)
        z = zeros(p * n_active)
        apply_J!(y, stamp, v)
        apply_JT!(z, stamp, u, trues(n_active))
        @test dot(u, y) ≈ dot(z, v) rtol = 1e-12
    end

    @testset "H accumulation matches dense J'WJ" begin
        p = 3
        S2 = 9
        n_active = 6
        npix = 30
        pixels = zeros(Int32, S2, n_active)
        for a in 1:n_active
            for m in 1:S2
                fi = mod(m + (a - 1) * 2 - 1, npix) + 1
                pixels[m, a] = fi
            end
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        union_pix, nbr_blocks = _build_neighbors(pixels, S2, n_active, p, Float64)
        Hdiag = zeros(p, p, n_active)
        _accumulate_H!(Hdiag, nbr_blocks, stamp)

        J = dense_J(stamp)
        H_dense = J' * J
        H_acc = assemble_H_dense(Hdiag, nbr_blocks, p, n_active)
        @test H_acc ≈ H_dense rtol = 1e-10 atol = 1e-12
        # diag(H_scaled) == 1 only holds for equilibrated values; check the
        # block agreement directly instead.
    end

    @testset "PCG matches dense damped solve" begin
        p = 3
        S2 = 9
        n_active = 6
        npix = 8 * n_active + 8
        pixels = zeros(Int32, S2, n_active)
        # Light overlap (1 shared pixel between adjacent stars) so H is
        # well-conditioned and PCG converges tightly.
        for a in 1:n_active
            for m in 1:S2
                fi = mod(m + (a - 1) * 8 - 1, npix) + 1
                pixels[m, a] = fi
            end
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        equilibrate!(stamp)
        _, nbr_blocks = _build_neighbors(pixels, S2, n_active, p, Float64)
        Hdiag = zeros(p, p, n_active)
        _accumulate_H!(Hdiag, nbr_blocks, stamp)
        H = assemble_H_dense(Hdiag, nbr_blocks, p, n_active)
        H = Symmetric(H)  # accumulate produces the full symmetric block structure

        λ = 1.0e-3
        n = p * n_active
        rhs = randn(rng, n)
        δ_dense = (Matrix(H) + λ * Matrix{Float64}(I, n, n)) \ rhs

        Mblocks = zeros(p, p, n_active)
        _build_precond!(Mblocks, Hdiag, λ)
        δ = zeros(n)
        _pcg!(δ, Hdiag, nbr_blocks, λ, rhs, Mblocks, n,
            zeros(n), zeros(n), zeros(n), zeros(n), Int[], p, 0.0)
        @test δ ≈ δ_dense rtol = 1e-8 atol = 1e-10
    end

    @testset "Cholesky cache matches dense damped solve" begin
        p = 3
        S2 = 9
        n_active = 6
        npix = 30
        pixels = zeros(Int32, S2, n_active)
        for a in 1:n_active
            for m in 1:S2
                fi = mod(m + (a - 1) * 2 - 1, npix) + 1
                pixels[m, a] = fi
            end
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        _, nbr_blocks = _build_neighbors(pixels, S2, n_active, p, Float64)
        Hdiag = zeros(p, p, n_active)
        _accumulate_H!(Hdiag, nbr_blocks, stamp)
        H1 = assemble_H_dense(Hdiag, nbr_blocks, p, n_active)

        cache = _build_cholesky_cache(nbr_blocks, n_active, p)
        live_all = trues(n_active)
        _refill_cholesky!(cache, Hdiag, nbr_blocks, live_all)

        n = p * n_active
        rhs = randn(rng, n)
        λ1 = 1.0e-3
        δ1_dense = (H1 + λ1 * Matrix{Float64}(I, n, n)) \ rhs
        δ1 = zeros(n)
        _solve_cholesky!(δ1, cache, λ1, rhs)
        @test δ1 ≈ δ1_dense rtol = 1e-8 atol = 1e-10

        # A second linearization: new stamp values, refill in place, reuse F.
        stamp2 = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        _accumulate_H!(Hdiag, nbr_blocks, stamp2)
        H2 = assemble_H_dense(Hdiag, nbr_blocks, p, n_active)
        _refill_cholesky!(cache, Hdiag, nbr_blocks, live_all)
        λ2 = 5.0e-2
        δ2_dense = (H2 + λ2 * Matrix{Float64}(I, n, n)) \ rhs
        δ2 = zeros(n)
        _solve_cholesky!(δ2, cache, λ2, rhs)
        @test δ2 ≈ δ2_dense rtol = 1e-8 atol = 1e-10
    end

    @testset "_refill_cholesky! zeroes frozen-touching pairs" begin
        p = 3
        S2 = 9
        n_active = 6
        npix = 30
        pixels = zeros(Int32, S2, n_active)
        for a in 1:n_active
            for m in 1:S2
                fi = mod(m + (a - 1) * 2 - 1, npix) + 1
                pixels[m, a] = fi
            end
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        _, nbr_blocks = _build_neighbors(pixels, S2, n_active, p, Float64)
        Hdiag = zeros(p, p, n_active)
        _accumulate_H!(Hdiag, nbr_blocks, stamp)
        @test length(nbr_blocks) > 0  # this fixture must actually have pairs

        cache = _build_cholesky_cache(nbr_blocks, n_active, p)
        live = trues(n_active)
        live[3] = false  # freeze star 3 (1-based)
        _refill_cholesky!(cache, Hdiag, nbr_blocks, live)

        for t in 1:length(nbr_blocks)
            a = nbr_blocks.pair_a[t] + 1
            b = nbr_blocks.pair_b[t] + 1
            frozen_pair = a == 3 || b == 3
            for k in 1:p, l in 1:p
                pos = cache.invord[cache.pair_pos[k, l, t]]
                if frozen_pair
                    @test cache.H.nzval[pos] == 0.0
                else
                    @test cache.H.nzval[pos] == nbr_blocks.B[k, l, t]
                end
            end
        end
    end

    @testset "_pcg! masking keeps frozen block exactly zero" begin
        p = 3
        S2 = 9
        n_active = 3
        npix = 8 * n_active + 8
        pixels = zeros(Int32, S2, n_active)
        for a in 1:n_active
            for m in 1:S2
                fi = mod(m + (a - 1) * 8 - 1, npix) + 1
                pixels[m, a] = fi
            end
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        equilibrate!(stamp)
        _, nbr_blocks = _build_neighbors(pixels, S2, n_active, p, Float64)
        @test length(nbr_blocks) > 0
        Hdiag = zeros(p, p, n_active)
        _accumulate_H!(Hdiag, nbr_blocks, stamp)

        λ = 1.0e-3
        n = p * n_active
        rhs = randn(rng, n)
        rhs[(3 - 1) * p + 1:3 * p] .= 0.0  # star 3 frozen -> rhs slice is 0

        Mblocks = zeros(p, p, n_active)
        _build_precond!(Mblocks, Hdiag, λ)

        # Masked: frozen_list = [2] (0-based index of star 3).
        δ_masked = zeros(n)
        _pcg!(δ_masked, Hdiag, nbr_blocks, λ, rhs, Mblocks, n,
            zeros(n), zeros(n), zeros(n), zeros(n), [2], p, 0.0)
        @test all(iszero, view(δ_masked, (3 - 1) * p + 1:3 * p))

        # Unmasked (frozen_list = []): if star 3 has any live neighbor with a
        # nonzero pair block, its slice should NOT stay zero -- this is the
        # negative control demonstrating the leakage the mask prevents.
        δ_unmasked = zeros(n)
        _pcg!(δ_unmasked, Hdiag, nbr_blocks, λ, rhs, Mblocks, n,
            zeros(n), zeros(n), zeros(n), zeros(n), Int[], p, 0.0)
        @test any(!iszero, view(δ_unmasked, (3 - 1) * p + 1:3 * p))
    end

    @testset "_pcg! cg_tol stops early without sacrificing accuracy" begin
        p = 3
        S2 = 9
        n_active = 6
        npix = 8 * n_active + 8
        pixels = zeros(Int32, S2, n_active)
        for a in 1:n_active
            for m in 1:S2
                fi = mod(m + (a - 1) * 8 - 1, npix) + 1
                pixels[m, a] = fi
            end
        end
        stamp = StampDerivatives{Float64, Int32}(
            randn(rng, p, S2, n_active), pixels, zeros(p, n_active), npix, p, S2)
        equilibrate!(stamp)
        _, nbr_blocks = _build_neighbors(pixels, S2, n_active, p, Float64)
        Hdiag = zeros(p, p, n_active)
        _accumulate_H!(Hdiag, nbr_blocks, stamp)

        λ = 1.0e-3
        n = p * n_active
        rhs = randn(rng, n)
        δ_dense = (Matrix(assemble_H_dense(Hdiag, nbr_blocks, p, n_active)) +
            λ * Matrix{Float64}(I, n, n)) \ rhs

        Mblocks = zeros(p, p, n_active)
        _build_precond!(Mblocks, Hdiag, λ)

        # A loose cg_tol with a generous iteration cap should still recover
        # the accurate dense solution -- the residual check, not the cap, is
        # what's doing the stopping.
        δ = zeros(n)
        _pcg!(δ, Hdiag, nbr_blocks, λ, rhs, Mblocks, n,
            zeros(n), zeros(n), zeros(n), zeros(n), Int[], p, 1e-10)
        @test δ ≈ δ_dense rtol = 1e-6 atol = 1e-9

        # A cap far below what a fixed-iteration solve would need still
        # gives an accurate answer once cg_tol is satisfied, as long as the
        # cap isn't reached first.
        δ_capped = zeros(n)
        _pcg!(δ_capped, Hdiag, nbr_blocks, λ, rhs, Mblocks, 2 * n,
            zeros(n), zeros(n), zeros(n), zeros(n), Int[], p, 1e-10)
        @test δ_capped ≈ δ_dense rtol = 1e-6 atol = 1e-9

        # A tiny cap below what's needed to reach cg_tol leaves a real,
        # detectable residual -- confirms the cap still binds when it must.
        δ_undersolved = zeros(n)
        _pcg!(δ_undersolved, Hdiag, nbr_blocks, λ, rhs, Mblocks, 1,
            zeros(n), zeros(n), zeros(n), zeros(n), Int[], p, 1e-10)
        @test !isapprox(δ_undersolved, δ_dense; rtol = 1e-6, atol = 1e-9)
    end

    @testset "_compact_neighbors preserves indices and structure" begin
        p = 3
        S2 = 9
        n_active = 8
        npix = 40
        pixels = zeros(Int32, S2, n_active)
        for a in 1:n_active
            for m in 1:S2
                fi = mod(m + (a - 1) * 2 - 1, npix) + 1
                pixels[m, a] = fi
            end
        end
        _, nbr_blocks = _build_neighbors(pixels, S2, n_active, p, Float64)
        @test length(nbr_blocks) > 0

        live = trues(n_active)
        live[2] = false
        live[5] = false
        compacted = _compact_neighbors(nbr_blocks, live, p)

        expected_keep = count(1:length(nbr_blocks)) do t
            live[nbr_blocks.pair_a[t] + 1] && live[nbr_blocks.pair_b[t] + 1]
        end
        @test length(compacted) == expected_keep
        @test all(live[compacted.pair_a[t] + 1] && live[compacted.pair_b[t] + 1] for t in 1:length(compacted))
        # Star indices are not renumbered: every surviving pair's (a, b) must
        # appear, unchanged, among the original pairs.
        orig_pairs = Set((nbr_blocks.pair_a[t], nbr_blocks.pair_b[t]) for t in 1:length(nbr_blocks))
        @test all((compacted.pair_a[t], compacted.pair_b[t]) in orig_pairs for t in 1:length(compacted))
        @test compacted.offsets[end] - 1 == length(compacted.shared_ma)

        # Freezing everyone matches _build_neighbors's empty-pair convention.
        compacted_empty = _compact_neighbors(nbr_blocks, falses(n_active), p)
        @test length(compacted_empty) == 0
        @test isempty(compacted_empty.offsets)
    end

    @testset "isolated star matches fit_all_stars" begin
        circ_psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        # Sample the circular Gaussian onto a unit-oversampled grid and wrap it
        # as a (single-node) GriddedPSFModel, so the loop below also exercises
        # `_fill_stamps!`/`_render_model!`'s specialized
        # `GriddedPSFModel{T,<:ImagePSF{T}}` methods against the same fixture.
        node_data = render(circ_psf)
        node = ImagePSF(node_data; y=0.0, x=0.0, flux=1.0, bkg=0.0, oversampling=1, normalize=true)
        gridded_psf = GriddedPSFModel([node], [0.0], [0.0]; y=0.0, x=0.0, flux=1.0, bkg=0.0)

        for (psf, fixed) in ((circ_psf, (; fwhm=2.0, bkg=0.0)), (gridded_psf, (; bkg=0.0)))
            rng2 = StableRNG(42)
            image, sources = simulate_image((128, 128), psf, 5;
                background=20.0, noise=:none, flux=(600.0, 900.0),
                min_separation=15, border=10, model_radius=5, rng=rng2)
            img_sub = image .- 20.0
            cat = (; y=sources.y, x=sources.x, flux=fill(400.0, 5))
            r_seq = fit_all_stars(img_sub, psf, cat, 5; fixed, n_passes=1, max_iter=200)
            # This fixture checks convergence to near machine precision in the
            # noiseless limit, which needs a tighter `x_tol` than
            # `fit_all_stars_simultaneous`'s default which is appropriate for noisy data.
            r_sim = fit_all_stars_simultaneous(img_sub, psf, cat, 5;
                fixed, max_iter=40, inner_iterations=10, x_tol=1e-8)
            @test all(r_sim.valid)
            @test r_sim.flux ≈ r_seq.flux rtol = 1e-10
            @test r_sim.y ≈ r_seq.y atol = 1e-10
            @test r_sim.x ≈ r_seq.x atol = 1e-10
            @test r_sim.flux ≈ sources.flux rtol = 1e-10
            # The cholesky oracle agrees with the default CG solver.
            r_chol = fit_all_stars_simultaneous(img_sub, psf, cat, 5;
                fixed, solver = :cholesky, max_iter = 40, x_tol=1e-8)
            @test r_chol.flux ≈ r_sim.flux rtol = 1e-10
            @test r_chol.y ≈ r_sim.y atol = 1e-10
        end
    end

    @testset "Cholesky/CG agree in a crowded fit that forces mid-fit freezing" begin
        # Unlike the isolated-star fixture above, this one is crowded enough
        # that most stars freeze well before max_iter -- it exercises _pcg!'s
        # Ap masking + NeighborBlocks compaction (:cg) against
        # _refill_cholesky!'s zero-in-place masking (:cholesky) together,
        # not just the unfrozen path.
        rng7 = StableRNG(11)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((150, 150), psf, 200;
            background=50.0, noise=:poisson_gaussian, read_noise=2.0, gain=1.0,
            flux=(100.0, 3000.0), flux_distribution=:powerlaw, flux_power=2.0,
            min_separation=2, border=8, model_radius=6, rng=rng7)
        img_sub = image .- 50.0
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        fixed = (; fwhm=2.0, bkg=0.0)
        # Disable f_tol, tight `x_tol` (see the isolated-star fixture above for why):
        # the CG/Cholesky agreement tolerances are tighter than the
        # default, since a looser `x_tol`
        # lets the two solvers freeze the same star at slightly different
        # iterations in this degenerate, heavily-blended fixture.
        r_cg = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=25, x_tol=1e-8, f_tol=0)
        r_chol = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=25, solver=:cholesky, x_tol=1e-8, f_tol=0)
        # This fixture must actually exercise freezing before the fit ends.
        @test sum(r_cg.n_iter[r_cg.valid] .< r_cg.n_passes) > 0.5 * sum(r_cg.valid)
        g = r_cg.valid .& r_chol.valid
        @test sum(g) > 0.8 * length(g)
        @test r_cg.flux[g] ≈ r_chol.flux[g] rtol = 1e-2
        # Positions are compared per-star rather than with `≈ atol=...`, which
        # for arrays compares 2-norms and so fails as soon as *any* single star
        # in this deliberately degenerate fixture converges one iteration apart
        # between the two solvers.  The bulk must
        # agree far below the centroid noise floor, which is what a genuine
        # disagreement between `_pcg!`'s Ap masking / `NeighborBlocks`
        # compaction and `_refill_cholesky!`'s zero-in-place masking would
        # break.
        dy = abs.(r_cg.y[g] .- r_chol.y[g])
        dx = abs.(r_cg.x[g] .- r_chol.x[g])
        @test median(dy) < 1e-5
        @test median(dx) < 1e-5
        @test quantile(dy, 0.9) < 1e-3
        @test quantile(dx, 0.9) < 1e-3
        @test maximum(dy) < 2e-2
        @test maximum(dx) < 2e-2
    end

    @testset "free-set contract" begin
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((64, 64), psf, 1;
            background=20.0, noise=:none, flux=(500.0, 500.0),
            border=8, model_radius=5, rng=StableRNG(1))
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        # bkg left free -> fwhm also free -> error.
        @test_throws ArgumentError fit_all_stars_simultaneous(image, psf, cat, 5)
        # LevenbergDamping rejected.
        @test_throws ArgumentError fit_all_stars_simultaneous(image, psf, cat, 5;
            fixed=(; fwhm=2.0, bkg=0.0), damping=CrowdPhot.LevenbergDamping())
        # Bad solver.
        @test_throws ArgumentError fit_all_stars_simultaneous(image, psf, cat, 5;
            fixed=(; fwhm=2.0, bkg=0.0), solver=:foo)
    end

    @testset "empty input" begin
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        img = zeros(16, 16)
        cat = (; y=Float64[], x=Float64[], flux=Float64[])
        r = fit_all_stars_simultaneous(img, psf, cat, 5; fixed=(; fwhm=2.0, bkg=0.0))
        @test isempty(r.flux)
        @test r.n_passes == 0
    end

    @testset "failure paths" begin
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        img = zeros(32, 32)
        # Non-finite image, no inv_var -> error.
        img_bad = copy(img); img_bad[5, 5] = NaN
        @test_throws ArgumentError fit_all_stars_simultaneous(img_bad, psf,
            (; y=[16.0], x=[16.0], flux=[100.0]), 5; fixed=(; fwhm=2.0, bkg=0.0))
        # Fully masked star -> excluded before the solve.
        inv_var = ones(32, 32)
        inv_var[10:22, 10:22] .= 0.0
        r = fit_all_stars_simultaneous(img, psf,
            (; y=[16.0], x=[16.0], flux=[100.0]), 5;
            fixed=(; fwhm=2.0, bkg=0.0), inv_var)
        @test r.n_failed == 1
        @test !r.valid[1]
    end

    @testset "qfit is global-residual based" begin
        # qfit equals sum(|global residual|) / flux
        rng4 = StableRNG(5)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((64, 64), psf, 3;
            background=20.0, noise=:none, flux=(500.0, 800.0),
            min_separation=6, border=8, model_radius=5, rng=rng4)
        img_sub = image .- 20.0
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        fixed = (; fwhm=2.0, bkg=0.0)
        r = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=40)
        for i in 1:length(r.flux)
            r.valid[i] || continue
            yr = floor(Int, r.y[i] - 5):ceil(Int, r.y[i] + 5)
            xr = floor(Int, r.x[i] - 5):ceil(Int, r.x[i] + 5)
            yr, xr = _clamp_inds(yr, xr, r.residual)
            @test r.qfit[i] ≈ sum(abs, view(r.residual, yr, xr)) / r.flux[i] rtol = 1e-10
        end
    end

    @testset "spread_model matches sequential path" begin
        rng8 = StableRNG(21)
        fwhm_psf = 2.5
        psf = CircularGaussianPSF(y = 0.0, x = 0.0, fwhm = fwhm_psf, flux = 1.0, bkg = 0.0)
        # isolated stars + one injected broadened Gaussian
        img = fill(20.0, (90, 90))
        pts = [(20.5, 20.5), (20.5, 68.5), (68.5, 20.5)]
        for (yy, xx) in pts
            CrowdPhot.PSF.add_star!(img, CircularGaussianPSF(y = yy, x = xx, fwhm = fwhm_psf, flux = 4000.0, bkg = 0.0))
        end
        CrowdPhot.PSF.add_star!(img, CircularGaussianPSF(y = 68.5, x = 68.5, fwhm = 5.0, flux = 4000.0, bkg = 0.0))
        img_sub = img .- 20.0
        cat = (; y = [first.(pts); 68.5], x = [last.(pts); 68.5], flux = fill(4000.0, 4))
        iv = fill(1 / 20.0, size(img))
        fixed = (; fwhm = fwhm_psf, bkg = 0.0)
        rseq = fit_all_stars(img_sub, psf, cat, 7; fixed, n_passes = 3, max_iter = 100, inv_var = iv)
        rsim = fit_all_stars_simultaneous(img_sub, psf, cat, 7; fixed, max_iter = 40, inv_var = iv)
        for i in 1:3  # the isolated point sources
            @test rseq.valid[i] && rsim.valid[i]
            @test isapprox(rseq.spread_model[i], 0.0; atol = 3e-3)
            @test isapprox(rsim.spread_model[i], 0.0; atol = 3e-3)
            @test isapprox(rseq.spread_model[i], rsim.spread_model[i]; atol = 2e-3)
        end
        # the extended source: positive in both, and consistent
        @test rseq.spread_model[4] > 5e-3
        @test rsim.spread_model[4] > 5e-3
        @test isapprox(rseq.spread_model[4], rsim.spread_model[4]; rtol = 0.25)
    end

    @testset "recovery vs truth" begin
        rng3 = StableRNG(7)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((160, 160), psf, 120;
            background=100.0, noise=:poisson_gaussian, read_noise=2.0, gain=1.5,
            flux=(30.0, 3000.0), flux_distribution=:powerlaw, flux_power=2.0,
            min_separation=2, border=10, model_radius=8, rng=rng3)
        img_sub = image .- 100.0
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        fixed = (; fwhm=2.0, bkg=0.0)
        r = fit_all_stars_simultaneous(img_sub, psf, cat, 5;
            fixed, max_iter=15, inner_iterations=10)
        g = r.valid
        @test sum(g) > 0.8 * length(g)
        # Bright sources recover flux more accurately than faint ones.
        f = sources.flux[g]
        bias = (r.flux[g] .- f) ./ f
        bright = f .> median(f)
        faint = .!bright
        @test median(abs.(bias[bright])) < median(abs.(bias[faint]))
    end

    @testset "per-star convergence is non-uniform" begin
        rng5 = StableRNG(99)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((200, 200), psf, 60;
            background=50.0, noise=:poisson_gaussian, read_noise=2.0, gain=1.0,
            flux=(200.0, 3000.0), flux_distribution=:powerlaw, flux_power=2.0,
            min_separation=2, border=10, model_radius=6, rng=rng5)
        img_sub = image .- 50.0
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        fixed = (; fwhm=2.0, bkg=0.0)
        r = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=30)
        g = r.valid
        @test sum(g) > 0.5 * length(g)
        # Not every star takes the whole loop to freeze, and not every star
        # freezes at the same iteration.
        @test any(r.n_iter[g] .< r.n_passes)
        @test length(unique(r.n_iter[g])) > 1
    end

    @testset "frozen star parameters are stable under extra iterations" begin
        rng6 = StableRNG(99)
        psf = CircularGaussianPSF(y=0.0, x=0.0, fwhm=2.0, flux=1.0, bkg=0.0)
        image, sources = simulate_image((200, 200), psf, 60;
            background=50.0, noise=:poisson_gaussian, read_noise=2.0, gain=1.0,
            flux=(200.0, 3000.0), flux_distribution=:powerlaw, flux_power=2.0,
            min_separation=2, border=10, model_radius=6, rng=rng6)
        img_sub = image .- 50.0
        cat = (; y=sources.y, x=sources.x, flux=copy(sources.flux))
        fixed = (; fwhm=2.0, bkg=0.0)
        K1, K2 = 10, 25
        r1 = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=K1)
        r2 = fit_all_stars_simultaneous(img_sub, psf, cat, 5; fixed, max_iter=K2)
        # Stars that froze at or before K1 in the longer run took an
        # identical path through the loop in the shorter run (nothing before
        # iteration K1 depends on max_iter), so their final values must
        # match exactly -- the direct operational check that freezing
        # actually stops a star's motion rather than merely reporting it as
        # converged.
        frozen_early = [i for i in eachindex(sources.y)
                        if r1.valid[i] && r2.valid[i] && 0 < r2.n_iter[i] <= K1]
        @test length(frozen_early) > 0  # fixture must actually exercise this
        for i in frozen_early
            @test r1.y[i] ≈ r2.y[i] atol = 1e-8
            @test r1.x[i] ≈ r2.x[i] atol = 1e-8
            @test r1.flux[i] ≈ r2.flux[i] rtol = 1e-8
        end
    end
end
