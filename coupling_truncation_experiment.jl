# Coupling-graph truncation experiment: why naive top-K pruning is unsafe,
# and what actually works.
# ============================================================================
#
# Context: neither exact block-level Takahashi selected inversion
# (block_takahashi_prototype.jl, ~5.8x faster than shipped, still 32.3s at
# the target scale) nor Hutchinson-style stochastic probing
# (hutchinson_probing_experiment.jl, negative result -- needs impractically
# many probes due to the huge dynamic range in per-star variances) clear the
# bar for the actual workload (100,000 stars, 2000x2000 image, fit_rad=5,
# avg_deg~11.2 neighbors/star, ~0.025 stars/px^2 -- and worse densities are
# expected). This experiment tests whether truncating the star-coupling
# graph used ONLY for the error/covariance step (keeping the full fit_rad
# stamps for the actual photometry) is a viable third option, per two ideas
# raised in conversation: (1) an adaptive radius that auto-scales to bound
# total Cholesky fill-in, and (2) keeping only each star's single most
# significant neighbor.
#
# ----------------------------------------------------------------------
# Finding 1 (important, negative): naive hard top-K pruning by coupling
# strength is NOT reliably positive-semi-definite-preserving
# ----------------------------------------------------------------------
# H = J'WJ (Gauss-Newton normal equations) is PSD as a WHOLE, being a sum of
# per-pixel rank-1 (or low-rank) outer-product contributions. But
# arbitrarily zeroing SOME off-diagonal p x p star-pair blocks while keeping
# others -- a purely combinatorial, post-hoc edit of an already-assembled H
# -- does NOT preserve that sum structure and is not guaranteed PSD.
#
# Empirical demonstration on the real 100,000-star field: built a per-pair
# "coupling strength" metric (Frobenius norm of the off-diagonal block,
# normalized by the geometric mean of the two stars' own diagonal-block
# norms -- a correlation-like quantity), then for each star kept only its
# top-K strongest-coupled neighbors, combining across stars two ways:
#   - "mutual" (AND): keep pair (a,b) only if each considers the other in
#     its own top-K
#   - "union" (OR): keep pair (a,b) if EITHER does
#
#   K=1 mutual (avg_deg=0.62):  SUCCEEDS at shift=1e-12 (production's
#                                actual first-attempt shift)
#   K=1 union  (avg_deg=1.38):  FAILS -- PosDefException even escalating
#                                shift up to 1e-2 (1e-12 * 10^10)
#   K=2,3,5,8 mutual:           ALL FAIL the same way, despite K=1 mutual
#                                succeeding -- MORE retained coupling made
#                                things WORSE, not better, which is the
#                                signature of a structural (not numerical)
#                                problem, not "just needs a bigger shift"
#
# K=1 mutual's success looks like luck (a small enough perturbation that
# diagonal dominance survived by accident for this field), not a property
# that should be relied on in general -- there is no guarantee it holds for
# a different density, PSF, or star configuration. DO NOT ship hard top-K
# pruning of an already-assembled H.
#
# ----------------------------------------------------------------------
# Finding 2: covariance tapering IS provably safe, but the naive
# parametrization tried here gave poor accuracy
# ----------------------------------------------------------------------
# The rigorous fix for "sparsify a PSD matrix without breaking PSD-ness" is
# covariance tapering (Furrer, Genton & Nychka 2006): multiply (Schur/
# Hadamard product) the matrix by a taper matrix built from a valid
# compactly-supported positive-definite radial kernel (e.g. Wendland 1995's
# C^2 function). Justification: let Theta be the n_active x n_active
# star-level taper matrix (Theta[a,b] = kernel(dist(a,b)/r), Theta[a,a]=1),
# and T = Theta ⊗ J_p (Kronecker product with the p x p all-ones matrix
# J_p, itself PSD as rank-1 = ones(p)*ones(p)'). Kronecker product of two
# PSD matrices is PSD, so T is PSD whenever Theta is (i.e. whenever the
# radial kernel is a valid positive-definite function in 2D, which Wendland
# functions are by construction). The Schur product theorem then gives:
# H .* T is PSD whenever H is, REGARDLESS of how many entries the taper
# drives to exactly zero. Concretely this means: multiply each off-diagonal
# block nb.B[:,:,t] by a single scalar taper(dist(a,b)/r) (T is block-
# constant, so every entry within block (a,b) gets the same scalar) --
# leave diagonal blocks untouched (taper(0)=1).
#
# Verified empirically: ALL r_taper values tested (2, 3, 4, 5, 6, 8) on the
# real 100,000-star field succeeded at shift=1e-12 with NO escalation
# needed -- in sharp contrast to Finding 1. This part of the theory is
# confirmed correct and safe.
#
# BUT accuracy was poor and barely improved across a 16x change in retained
# pairs:
#   r_taper   avg_deg   flux relerr (median/90th)   y relerr (median/90th)
#      2       0.32          0.169 / 0.994               0.180 / 0.935
#      3       0.72          0.169 / 0.993               0.180 / 0.930
#      4       1.28          0.169 / 0.991               0.179 / 0.922
#      5       2.00          0.168 / 0.989               0.178 / 0.913
#      6       2.88          0.165 / 0.986               0.173 / 0.903
#      8       5.12          0.153 / 0.979               0.159 / 0.885
# (median pair distance in this field was 8.36 px; wendland_taper(d>=1)=0,
# so r_taper=8 already drops roughly half of all nominal "neighbor" pairs,
# and even RETAINED pairs get substantially damped -- e.g. a pair at half
# the cutoff distance is already multiplied by only 0.19.)
#
# Compare to Finding 1's K=1 MUTUAL result (avg_deg=0.62, similar sparsity
# to r_taper=2-3): flux median 0.6%, y median 1.3% -- an order of magnitude
# better than ANY tapering radius tried, despite retaining FEWER pairs.
# This strongly suggests raw Euclidean distance is a poor proxy for which
# couplings actually matter here (most fit_rad-driven "neighbor" pairs at
# this density sit near the outer edge of stamp overlap, distance ~8px for
# fwhm=2.5, i.e. deep in the Gaussian PSF's negligible-overlap tail -- true
# variance-relevant coupling is concentrated in a much smaller population
# of genuinely close/blended pairs that a fixed-shape distance taper
# doesn't isolate well), while coupling-strength ranking (Finding 1) does
# identify the right pairs but isn't safely sparsifiable post-hoc.
#
# ----------------------------------------------------------------------
# RECOMMENDATION (not yet implemented): don't post-hoc prune an assembled
# H at all -- adapt the RADIUS FED INTO the existing H-assembly pipeline
# ----------------------------------------------------------------------
# _build_stamps!/_fill_stamps!/_accumulate_H! already build H correctly (by
# construction PSD, being a sum of per-pixel outer products) for whatever
# radius they're given. Re-running that SAME pipeline with a smaller
# "error radius" r_err <= fit_rad -- computed ONLY for the final
# covariance step, after the fit has already converged using the full
# fit_rad -- sidesteps Finding 1's PSD-breaking risk entirely: it's exactly
# the same assembly as the (already correct, tested) fit_rad=r_err case,
# just with fewer pixels/pairs contributing. This directly implements the
# "auto-scaling adaptive radius" idea without needing tapering's Schur-
# product cleverness (or its apparent accuracy problem) at all.
#
# How to choose r_err adaptively: this session's fit_rad sweep already
# measured avg_deg (= 2*npairs/n_active) as a function of (density, r) at
# several points (see block_takahashi_prototype.jl's status notes and
# earlier conversation), e.g. avg_deg ~ rho * r^2 up to a geometric
# constant. Given a computational budget (target average degree, or a
# directly-measured flop-proxy/nnz(L) budget calibrated against the
# already-collected timing data), solve for the largest r_err consistent
# with that budget using the local (or global) star density rho. This
# would need: (a) a quick way to estimate rho (global n_active/area, or a
# local KD-tree-based estimate per region for spatially clumpy fields),
# and (b) picking a target average degree/fill budget informed by the
# timing curves already measured (e.g. the block recursion's ~30s wall for
# avg_deg~11 vs its sub-second cost for avg_deg~2-3 in the smaller test
# field).
#
# NOT YET DONE: implementing this adaptive-radius selection, or measuring
# its actual accuracy/speed tradeoff on the real field. That is the
# concrete next step if this direction is pursued.

using SparseArrays, LinearAlgebra, Statistics
using CrowdPhot: NeighborBlocks, _build_cholesky_cache, _refill_cholesky!

wendland_taper(d) = d >= 1 ? 0.0 : (1 - d)^4 * (1 + 4d)

"""
    build_tapered_nb(nb, dist_pair, r_taper) -> (NeighborBlocks, nkept)

Schur-product-safe sparsification of `nb`'s off-diagonal blocks: scales
each pair's block by `wendland_taper(dist_pair[t] / r_taper)`, dropping
pairs where the taper is exactly zero. See module docstring for the
positive-definiteness argument and the (disappointing, as tested here)
accuracy result.
"""
function build_tapered_nb(nb::NeighborBlocks, dist_pair::AbstractVector, r_taper::Real)
    npair = length(nb)
    taper = wendland_taper.(dist_pair ./ r_taper)
    idx = findall(>(0), taper)
    Bnew = copy(nb.B[:, :, idx])
    for (i, t) in enumerate(idx)
        Bnew[:, :, i] .*= taper[t]
    end
    return NeighborBlocks{Float64}(nb.pair_a[idx], nb.pair_b[idx], Int[], Int32[], Int32[], Bnew), length(idx)
end

"""
    topk_sets(adj, K) -> Vector{Set{Int}}

For each star, the set of its `K` strongest-coupled neighbors (by whatever
per-pair strength metric was used to build `adj`, a per-star adjacency
list of `(neighbor, strength, pair_index)` tuples).
"""
function topk_sets(adj, K::Int)
    n = length(adj)
    sets = Vector{Set{Int}}(undef, n)
    for a in 1:n
        lst = adj[a]
        if length(lst) <= K
            sets[a] = Set(first.(lst))
        else
            sorted = sort(lst; by = x -> -x[2])
            sets[a] = Set(first(t) for t in sorted[1:K])
        end
    end
    return sets
end

"""
    build_reduced_nb(nb, sets_top, rule) -> (NeighborBlocks, nkept)

Hard top-K pruning (`rule` is `:mutual` or `:union` -- see module docstring
Finding 1). NOT SAFE: demonstrated to produce a non-PSD reduced H for most
tested K/rule combinations on the real field. Kept here for reproducing
that finding, not for reuse.
"""
function build_reduced_nb(nb::NeighborBlocks, sets_top, rule::Symbol)
    npair = length(nb)
    keep = falses(npair)
    for t in 1:npair
        a = nb.pair_a[t] + 1
        b = nb.pair_b[t] + 1
        in_a = b in sets_top[a]
        in_b = a in sets_top[b]
        keep[t] = rule === :mutual ? (in_a && in_b) : (in_a || in_b)
    end
    idx = findall(keep)
    return NeighborBlocks{Float64}(nb.pair_a[idx], nb.pair_b[idx], Int[], Int32[], Int32[], nb.B[:, :, idx]), length(idx)
end
