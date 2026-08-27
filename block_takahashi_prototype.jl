# Block-level Takahashi selected-inversion prototype
# ====================================================
#
# STATUS: validated, NOT yet wired into src/. Superseded (for now) by a
# Hutchinson-style stochastic probing experiment, since even this block
# version is too slow at the actual target scale (100,000 stars, 2000x2000
# image, fit_rad=5, avg_deg~11.2 neighbors/star): 32.3s (1.5s cholesky +
# 30.8s block recursion) vs. 187.5s for the current shipped scalar hybrid
# implementation (src/sparse_inverse.jl's _selected_inverse_permuted). Both
# are considered "unacceptably slow" for the real workload. Keep this file
# around in case the coupling-radius-capping approach (see below) or a
# hybrid block+probing approach is revisited later.
#
# ------------------------------------------------------------------------
# Background / motivation
# ------------------------------------------------------------------------
# The existing selected_inverse machinery (src/sparse_inverse.jl) operates
# on the SCALAR sparse Cholesky factor L of the joint Gauss-Newton Hessian
# H (size n = p*n_active, p = 3 for y/x/flux). Even after fixing the
# generic-SparseMatrixCSC-getindex overhead (see git history: "scalar hybrid
# indexing fix"), the recursion still does O(sum_j d_j^2) SCALAR operations,
# where d_j is the post-fill-in degree of column j. This wastes a factor of
# ~p^2=9 versus operating directly on the natural p x p star-blocks that
# Hdiag/NeighborBlocks already use elsewhere in
# src/photometry/psf_photometry_simultaneous.jl.
#
# ------------------------------------------------------------------------
# Key idea: force a block-respecting permutation
# ------------------------------------------------------------------------
# Julia's `cholesky(A::SparseMatrixCSC; perm=...)` accepts a caller-supplied
# permutation instead of computing AMD internally. By computing AMD on the
# much smaller n_active-sized STAR adjacency graph (from NeighborBlocks)
# rather than the full n=p*n_active scalar graph, and expanding that block
# order into a scalar permutation that places each star's p rows
# contiguously, the resulting Cholesky factor L is PROVABLY block-uniform:
# every p x p sub-block of L is either fully dense or fully absent. This
# is a direct consequence of Hdiag's diagonal blocks being fully dense
# (each star's own p rows are mutually maximally connected before any
# elimination happens), which is exactly the classical definition of a
# Cholesky supernode. Verified empirically: 0 mismatches out of 8000
# sub-columns checked on a real 4000-star field, and fill quality
# (nnz(L)) within 0.03% of CHOLMOD's own default (unconstrained) AMD
# ordering -- forcing the block structure costs essentially nothing.
#
# ------------------------------------------------------------------------
# Block Takahashi recursion (derivation)
# ------------------------------------------------------------------------
# Starting from A = L L^T (block Cholesky, L_jj lower-triangular p x p,
# L_ij (i>j) general p x p), and Sigma = A^{-1}:
#   Sigma * L = L^{-T}   (since Sigma * L * L^T = I)
# Block column j, row i > j:
#   sum_{k>=j} Sigma_{i,k} L_{k,j} = 0
#   => Sigma_{i,j} = -[sum_{k>j} Sigma_{i,k} L_{k,j}] * L_{j,j}^{-1}
# Block column j, row i = j:
#   sum_{k>=j} Sigma_{j,k} L_{k,j} = L_{j,j}^{-T}
#   => Sigma_{j,j} = [L_{j,j}^{-T} - sum_{k>j} Sigma_{j,k} L_{k,j}] * L_{j,j}^{-1}
# where Sigma_{j,k} for k>j is Sigma_{k,j}^T (NOTE: this transpose was the
# first bug found during validation -- easy to drop by analogy with the
# scalar case, where transposition is invisible).
#
# Processing block columns j = n_active downto 1 (mirroring the scalar
# recursion's column order), all Sigma_{i,k} needed on the right-hand side
# (i,k both > j) are already known from earlier (larger-index) iterations.
#
# ------------------------------------------------------------------------
# Validation performed
# ------------------------------------------------------------------------
# - 2-star and 3-star synthetic fields: exact agreement with dense inverse
#   (~1e-13 absolute).
# - Full 4000-star random field (density 0.0111/px^2), fit_rad = 3..8:
#   RELATIVE agreement with the existing (correct, tested)
#   selected_inverse_diagonal_blocks at production's actual first-attempt
#   shift (1e-12): max relative diff ~5e-8, ZERO stars over 1e-6 relative
#   tolerance. (An earlier ABSOLUTE-difference comparison looked alarming
#   for ~29/4000 stars, but those turned out to be genuinely
#   near-singular/poorly-constrained stars where BOTH the scalar reference
#   and the block result are enormous and equally shift-sensitive --
#   confirmed by showing the scalar reference itself swings by ~2400x
#   between shift=0 and shift=1e-6 for the worst star. Not a bug; a
#   pre-existing numerical-conditioning sensitivity that affects both
#   implementations equally and is a separate concern from this
#   optimization -- see note at bottom.)
#
# ------------------------------------------------------------------------
# Performance measured (4000 stars, 600x600, density 0.0111/px^2)
# ------------------------------------------------------------------------
#   fit_rad   scalar (current)   block (this file)   speedup
#      5         0.034s              0.011s            3.1x
#      6         0.082s              0.030s            2.7x
#      7         0.301s              0.092s            3.3x
#      8         0.638s              0.259s            2.5x
#
# ------------------------------------------------------------------------
# Performance at the ACTUAL target scale (100,000 stars, 2000x2000,
# fit_rad=5, avg_deg~11.2 -- density 0.025/px^2, more than double the test
# field above)
# ------------------------------------------------------------------------
#   scalar (current shipped):  187.5s
#   block (this file):          32.3s  (1.5s cholesky + 30.8s recursion)
#   speedup: ~5.8x -- real, but still far too slow for interactive use.
#
# Root cause of the worse-than-linear scaling at this density: nnz(L) grew
# 34% faster than pure N-scaling would predict (29.26M vs the ~21.8M a
# linear-in-N extrapolation from the 4000-star/fit_rad=8 test would give),
# meaning fill-in itself stops scaling linearly with N once local coupling
# gets this dense -- squeezing the *exact* selected-inversion approach
# further has diminishing returns; this is fill-in, not implementation
# overhead.
#
# ------------------------------------------------------------------------
# Next directions considered (see conversation for full discussion)
# ------------------------------------------------------------------------
# 1. Cap the coupling radius used ONLY for the covariance/error step below
#    the fit's own fit_rad (keep fit_rad=5 stamps for photometry, use a
#    smaller neighbor graph for error propagation). Still exact math on a
#    smaller graph; user-facing accuracy/speed tradeoff.
# 2. Hutchinson-style stochastic diagonal-block probing using the existing
#    PCG matvec machinery (_apply_H!, _build_precond!) -- cost independent
#    of fill-in entirely. THIS IS THE DIRECTION BEING PURSUED NEXT (see
#    hutchinson_probing_experiment.jl / conversation for results).
#
# ------------------------------------------------------------------------
# Separate finding (not fixed here): near-singular star error-bar
# instability
# ------------------------------------------------------------------------
# For very poorly-constrained stars (e.g. a near-degenerate flux direction,
# Hdiag condition number ~1e4-1e5), fit_all_stars_simultaneous's reported
# flux_err is EXTREMELY sensitive to the regularizing `shift` -- observed a
# ~2400x swing in one star's reported variance between shift=0 and
# shift=1e-6, while `cholesky` reports success (no PosDefException) at
# shift=0, meaning the existing escalating-shift retry loop
# (psf_photometry_simultaneous.jl, `shifts = 1e-12 .* 10 .^ (0:6)`) never
# even triggers for this case since it only escalates on PosDefException,
# not on ill-conditioning short of outright singularity. This affects the
# CURRENT shipped scalar implementation too (verified: the scalar reference
# itself is what swings by ~2400x) -- it is not introduced by any of this
# session's changes. Worth a separate investigation/fix (e.g. a minimum
# shift regardless of PosDefException, or flagging/NaN-ing reported errors
# for stars above a condition-number threshold) but out of scope here.

using SparseArrays, LinearAlgebra, StaticArrays

"""
    _block_permutation(nb, n_active) -> Vector{Int}

Fill-reducing block permutation: AMD applied to the small n_active-sized
star-adjacency graph derived from `nb::NeighborBlocks` (not the full
n=p*n_active scalar graph). `shift` is set large enough to guarantee the
tiny structure-only adjacency matrix is SPD (values are irrelevant --
only used to extract `F.p`).
"""
function _block_permutation(nb, n_active::Int)
    npair = length(nb)
    I_ = Vector{Int}(undef, npair + n_active)
    J_ = Vector{Int}(undef, npair + n_active)
    for t in 1:npair
        I_[t] = nb.pair_a[t] + 1
        J_[t] = nb.pair_b[t] + 1
    end
    for a in 1:n_active
        I_[npair + a] = a
        J_[npair + a] = a
    end
    Ablock = sparse(I_, J_, ones(npair + n_active), n_active, n_active)
    Ablock = Ablock + Ablock' - Diagonal(diag(Ablock))
    Fblock = cholesky(Symmetric(Ablock, :L); shift = Float64(n_active))
    return Fblock.p
end

"""
    _block_selected_inverse_diagonal(Lsp::SparseMatrixCSC{T}, n_active, ::Val{P}) where {T,P} -> Vector{SMatrix{P,P,T}}

Block Takahashi recursion (see derivation above). `Lsp` must come from a
Cholesky factorization using a block-respecting permutation (see
`_block_permutation`), i.e. `sparse(cholesky(Symmetric(H,:L); perm=sperm).L)`
with `sperm` built by expanding `_block_permutation`'s output. Returns
diagonal covariance blocks in PERMUTED block order -- caller must map
`Diag[j]` back to original star index `bperm[j]`.

`browval` for each block-column is built from the UNION of all `P` scalar
columns in that block, not just the first -- a fill position can be
numerically canceled to near-zero in one column while remaining
structurally present (and non-negligible) in another column of the same
block; using only one column risks the position lookup silently running
past the end of that column's range. `findpos_bounded` errors loudly
instead of corrupting if this invariant is ever violated for a new input.
"""
function _block_selected_inverse_diagonal(Lsp::SparseMatrixCSC{T}, n_active::Int, ::Val{P}) where {T,P}
    rows = Lsp.rowval
    vals = Lsp.nzval

    bcolptr = Vector{Int}(undef, n_active + 1)
    bcolptr[1] = 1
    tmp_rowsets = Vector{Vector{Int32}}(undef, n_active)
    @inbounds for j in 1:n_active
        c1 = (j - 1) * P + 1
        s = Set{Int32}()
        for kk in 0:(P - 1)
            for idx in nzrange(Lsp, c1 + kk)
                push!(s, Int32((rows[idx] - 1) ÷ P + 1))
            end
        end
        seen = sort!(collect(s))
        tmp_rowsets[j] = seen
        bcolptr[j + 1] = bcolptr[j] + length(seen)
    end
    nblk = bcolptr[end] - 1
    browval = Vector{Int32}(undef, nblk)
    @inbounds for j in 1:n_active
        browval[bcolptr[j]:bcolptr[j + 1] - 1] .= tmp_rowsets[j]
    end

    @inline function findpos_bounded(lo, hi, b)
        pos = lo
        @inbounds while pos <= hi && browval[pos] != b
            pos += 1
        end
        pos > hi && error("block $b not found in expected column range -- fill pattern is not block-uniform")
        return pos
    end

    Lflat = zeros(T, P, P, nblk)
    @inbounds for j in 1:n_active
        c1 = (j - 1) * P + 1
        lo, hi = bcolptr[j], bcolptr[j + 1] - 1
        for kk in 1:P
            col = c1 + kk - 1
            for idx in nzrange(Lsp, col)
                r = rows[idx]
                b = Int32((r - 1) ÷ P + 1)
                pos = findpos_bounded(lo, hi, b)
                Lflat[(r - 1) % P + 1, kk, pos] = vals[idx]
            end
        end
    end
    Lb = [SMatrix{P,P,T,P*P}(view(Lflat, :, :, t)) for t in 1:nblk]

    Sigb = Vector{SMatrix{P, P, T, P * P}}(undef, nblk)
    Diag = Vector{SMatrix{P, P, T, P * P}}(undef, n_active)

    @inline function findpos(j, b)
        return findpos_bounded(bcolptr[j], bcolptr[j + 1] - 1, b)
    end

    @inbounds for j in n_active:-1:1
        lo, hi = bcolptr[j], bcolptr[j + 1] - 1
        Ljj = LowerTriangular(Lb[lo])
        Ljj_inv = inv(Ljj)
        for pidx in (lo + 1):hi
            i = browval[pidx]
            S = zero(MMatrix{P, P, T, P * P})
            for pidx2 in (lo + 1):hi
                k = browval[pidx2]
                Sik = i == k ? Diag[i] : (i > k ? Sigb[findpos(k, i)] : Sigb[findpos(i, k)]')
                S += Sik * Lb[pidx2]
            end
            Sigb[pidx] = (-SMatrix(S)) * Ljj_inv
        end
        Sjj = zero(MMatrix{P, P, T, P * P})
        for pidx2 in (lo + 1):hi
            k = browval[pidx2]
            Sjj += Sigb[pidx2]' * Lb[pidx2]
        end
        Diag[j] = (Ljj_inv' - SMatrix(Sjj)) * Ljj_inv
    end
    return Diag
end

"""
    block_selected_inverse_diagonal_blocks(H, nb, n_active, p; shift=0.0)

Convenience end-to-end wrapper: block-permute, factor, run the block
recursion, and depermute back to original star ordering. Drop-in
replacement for `selected_inverse_diagonal_blocks(H, p; shift)` (matches
its `(p,p,n_active)` output layout), but NOT currently wired into
src/photometry/psf_photometry_simultaneous.jl -- see status note at top.
"""
function block_selected_inverse_diagonal_blocks(H::SparseMatrixCSC{T}, nb, n_active::Int, p::Int; shift = zero(T)) where {T}
    bperm = _block_permutation(nb, n_active)
    sperm = Vector{Int}(undef, n_active * p)
    for j in 1:n_active, k in 1:p
        sperm[(j - 1) * p + k] = (bperm[j] - 1) * p + k
    end
    F = cholesky(Symmetric(H, :L); perm = sperm, shift = shift)
    Lsp = sparse(F.L)
    Diag_perm = _block_selected_inverse_diagonal(Lsp, n_active, Val(p))
    out = zeros(T, p, p, n_active)
    for j in 1:n_active
        out[:, :, bperm[j]] .= Diag_perm[j]
    end
    return out
end
