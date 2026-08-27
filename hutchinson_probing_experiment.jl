# Hutchinson-style stochastic probing experiment for marginal-variance
# estimation -- NEGATIVE RESULT, not recommended for adoption as-is.
# ============================================================================
#
# Context: block_takahashi_prototype.jl's exact block-level selected
# inversion, while ~5.8x faster than the shipped scalar implementation, is
# still too slow at the actual target scale (100,000 stars, 2000x2000 image,
# fit_rad=5, avg_deg~11.2 neighbors/star): 32.3s total (1.5s cholesky +
# 30.8s block recursion). This experiment tests whether a stochastic
# (Hutchinson) diagonal-block estimator -- cost independent of fill-in --
# could replace the exact recursion at competitive speed/accuracy.
#
# ----------------------------------------------------------------------
# Method
# ----------------------------------------------------------------------
# Standard Hutchinson estimator, generalized to full p x p blocks (not just
# scalar diagonal entries): for Rademacher probe vectors z (iid +-1 entries,
# independent across all n=p*n_active coordinates), and x = H^{-1} z,
#   E[x_i z_j] = (H^{-1})_{i,j}
# so accumulating outer products x[block] * z[block]' over many probes and
# averaging gives an unbiased estimate of each star's p x p marginal
# covariance block. Symmetrized ((S+S')/2) per block to reduce noise and
# enforce exact symmetry.
#
# Two ways to solve H x = z per probe were tried:
#  1. Block-Jacobi-preconditioned CG (reusing _apply_H!/_build_precond!/
#     _pcg! from psf_photometry_simultaneous.jl) -- REJECTED. At this
#     density (avg_deg~11.2), the block-Jacobi preconditioner (which only
#     sees each star's own p x p diagonal block, ignoring inter-star
#     coupling) is far too weak: with shift=1e-12 (production's actual
#     first-attempt regularization) CG diverges outright (residual grows to
#     ~2734x the initial value over 2000 iterations); even at shift=1e-2 it
#     needs 500-2000 iterations and 10-40s PER PROBE VECTOR -- already
#     worse than the entire exact block recursion for a single probe.
#  2. Direct sparse triangular solve via an already-computed Cholesky
#     factorization (`F \ z`, reusing CHOLMOD's supernodal forward/backward
#     substitution) -- USED BELOW. ~0.05-0.08s per probe at this scale
#     (measured, n=300,000), i.e. essentially free relative to the ~1.5s
#     one-time factorization cost. This is the only viable solve strategy
#     found; a from-scratch CG-based probing pass is not competitive here.
#
# ----------------------------------------------------------------------
# Result: accuracy vs. k (100,000-star field, fit_rad=5, density 0.025/px^2)
# ----------------------------------------------------------------------
# Ground truth: block_takahashi_prototype.jl's exact result (cross-checked
# against the shipped scalar implementation to <1.1e-7 max relative diff
# across all 100,000 stars -- both exact methods agree, so this is a solid
# reference).
#
#   k     time     flux-var relerr           y-var relerr
#               median    90th pct      median    90th pct
#   10    0.8s   0.0425    0.457         14.7      437
#   50    2.3s   0.0245    0.215          7.23     201
#   100   4.4s   0.0172    0.151          5.08     141
#   200   9.5s   0.0113    0.108          3.70     100
#   400  19.1s   0.00838   0.0751         2.57      71.2
#   800  36.1s   0.00581   0.0537         1.84      50.4
#
# Convergence follows the expected O(1/sqrt(k)) Monte Carlo rate exactly
# (an 80x increase in k from 10->800 reduces error by ~8x = sqrt(80) for
# both flux and y). The problem is the STARTING POINT and the SLOPE: y-variance
# relative error is still ~184% (median!) at k=800, which already costs as
# much wall time (36s) as the entire exact block computation (32.3s).
# Extrapolating the 1/sqrt(k) trend, reaching a merely-tolerable ~10% median
# y-variance error would require k ~ 270,000 -- about 3.4 HOURS at this
# per-probe cost. Not viable.
#
# ----------------------------------------------------------------------
# Why: dynamic range, not a bug
# ----------------------------------------------------------------------
# The true per-star marginal variances span an enormous dynamic range:
#   y-variance:    5.1e-7  to  3.8e5   (11 orders of magnitude)
#   flux-variance: 1.8e3   to  5.0e11  (8 orders of magnitude)
# The Hutchinson estimator's ABSOLUTE sampling noise is set by the overall
# scale/structure of H^{-1} and does not adapt per star. For the large
# population of well-constrained (tiny true variance) stars -- exactly
# the stars whose error bars are least in question -- that fixed noise
# floor completely swamps the true signal, producing enormous relative
# error. Flux fares somewhat better (narrower dynamic range, and the
# reported median is dominated by the bulk faint-star population where the
# relative noise floor happens to be more forgiving), but even flux's
# k=800 number (0.58% median) is already at cost-parity with the exact
# method, while its own tail (90th pct 5.4%) is still not great.
#
# ----------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------
# Plain per-star Hutchinson probing, as implemented here, is NOT a
# competitive replacement for exact selected inversion on this problem:
# by the time enough probes are used to get position variances to a usable
# accuracy, the wall-clock cost matches or exceeds the exact block
# recursion, which delivers EXACT results instead of noisy ones. This is a
# fundamentally different situation from typical Hutchinson use cases
# (e.g. estimating a single scalar trace tr(H^{-1}), where averaging over
# the whole diagonal lets errors partially cancel) -- here every individual
# per-star diagonal entry is needed, including many exponentially small
# ones, which is a much harder statistical target.
#
# Possible ways this could still be salvaged (NOT attempted, in increasing
# order of effort): (a) Hutch++-style deflation of a small number of
# dominant eigendirections before applying the stochastic estimator to the
# remainder (targets exactly the heavy-tailed dynamic range problem
# identified above, but requires a partial eigendecomposition/Lanczos
# pass); (b) variance-reduction via importance sampling or antithetic
# probe pairing; (c) applying probing ONLY to a subset of the coupling
# graph (e.g. within-clique probing rather than global). None of these
# were implemented or benchmarked here.
#
# RECOMMENDATION: do not pursue plain Hutchinson probing further for this
# use case. Revisit the other previously-identified lever (capping the
# coupling radius used for the error/covariance step below the fit's own
# fit_rad) instead.

using SparseArrays, LinearAlgebra, Random, Statistics

"""
    hutchinson_probe_direct(F, n, p, n_active, k; rng) -> Array{Float64,3}

Estimate all `n_active` marginal `p x p` covariance blocks of `H^{-1}`
(`F` a Cholesky factorization of `H`, or `H + shift*I`) via `k` Rademacher
probe vectors and direct sparse triangular solves (`F \\ z`). See the module
docstring above for why this direct-solve approach was used instead of an
iterative (CG) one, and for the negative accuracy-vs-cost result found.
"""
function hutchinson_probe_direct(F, n::Int, p::Int, n_active::Int, k::Int; rng = Random.default_rng())
    Sigma_est = zeros(Float64, p, p, n_active)
    z = Vector{Float64}(undef, n)
    for _ in 1:k
        @inbounds for i in 1:n
            z[i] = rand(rng, Bool) ? 1.0 : -1.0
        end
        x = F \ z
        @inbounds for a in 0:(n_active - 1)
            base = a * p
            for l in 1:p, kk in 1:p
                Sigma_est[kk, l, a + 1] += x[base + kk] * z[base + l]
            end
        end
    end
    Sigma_est ./= k
    @inbounds for a in 1:n_active
        blk = @view Sigma_est[:, :, a]
        blk .= (blk .+ blk') ./ 2
    end
    return Sigma_est
end
