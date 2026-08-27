# ==============================================================================
# Sparse selected inversion (Takahashi's recursion)
# ==============================================================================
#
# `selected_inverse` and `selected_inverse_diagonal_blocks` are the two
# entry points meant to be called directly; `_selected_inverse_permuted` and
# `_depermute_selected_inverse` are internal steps of `selected_inverse`
# only, not meant for standalone use.

"""
    _SELECTED_INVERSE_DICT_THRESHOLD

Column-degree cutoff used by [`_selected_inverse_permuted`](@ref) to decide,
per column, whether to precompute a `Dict`-based row-to-storage-position
lookup (worthwhile once a column's rows get probed often enough to amortize
building the `Dict`) or to fall back to a direct binary search over that
column's own row indices (cheaper for the many columns with only a handful
of entries, where building a `Dict` costs more than it saves). Chosen
empirically: below this, binary search wins; above it, the `Dict` wins, and
the win grows with column degree.
"""
const _SELECTED_INVERSE_DICT_THRESHOLD = 16

"""
    _selected_inverse_permuted(L::SparseArrays.SparseMatrixCSC{T}) where {T} -> SparseArrays.SparseMatrixCSC{T, Int}

Internal step of [`selected_inverse`](@ref): the core Takahashi
selected-inversion recursion (Takahashi et al. 1973; Erisman & Tinney 1975),
operating directly in the permuted coordinates of the Cholesky factor `L`
(so `A = L * L'` for the matrix `A` that `L` factors). Returns `Σ = inv(A)`
restricted to `L`'s own sparsity pattern: `Σ[i, j]` is only computed, and
only stored, for `i >= j` with `L[i, j]` structurally nonzero. Callers need
[`_depermute_selected_inverse`](@ref) to map the result back to the
original (unpermuted) row/column ordering.

Columns are processed in order `j = n, n-1, ..., 1`. For each column, every
off-diagonal entry `Σ[i, j]` (`i` ranging over `L`'s nonzero rows in column
`j`, excluding the diagonal) is computed from already-known `Σ` entries at
row/column pairs strictly greater than `j`. Those entries are guaranteed to
already be stored: Cholesky fill-in makes every pair of rows sharing a
column a clique in the filled graph, so any `(i, k)` pair that column `j`'s
recursion needs was itself part of some earlier (larger-index) column's
nonzero pattern. The diagonal entry `Σ[j, j]` is filled last, from the
freshly computed off-diagonal entries of column `j`.

Every write goes straight to a known position in `Σ.nzval`: `Σ` copies
`L`'s exact `(colptr, rowval)`, so a given loop index `idx` addresses the
same stored entry in both, and no search is needed to place a value. Every
off-diagonal *read* (`Σ[max(i, k), min(i, k)]`) resolves its position
through [`_SELECTED_INVERSE_DICT_THRESHOLD`](@ref)'s hybrid lookup instead
of `SparseMatrixCSC`'s generic `getindex`. Profiling showed that generic
`getindex`/`setindex!` path -- not the recursion's arithmetic -- dominates
wall time once fill-in (and hence average column degree) grows, e.g. with a
large `fit_rad` or a dense field; avoiding it recovers a large part of that
cost without changing the algorithm or its exactness.

# Arguments
- `L::SparseArrays.SparseMatrixCSC{T}`: sparse lower-triangular Cholesky
  factor (as returned by, e.g., `SparseArrays.sparse(cholesky(A).L)`).

# Returns
- `SparseArrays.SparseMatrixCSC{T, Int}` with the same sparsity pattern as
  `L`, holding `inv(A)` at every stored position.
"""
function _selected_inverse_permuted(L::SparseArrays.SparseMatrixCSC{T}) where {T}
    n = size(L, 1)
    rows = L.rowval
    vals = L.nzval
    colptr = L.colptr
    Σ = SparseArrays.SparseMatrixCSC(n, n, copy(colptr), copy(rows), zeros(T, SparseArrays.nnz(L)))
    Σnz = Σ.nzval

    # Per-column row->position lookup, built once up front: `Dict` for
    # columns large enough to amortize it, `nothing` (binary-search
    # fallback) otherwise. See `_SELECTED_INVERSE_DICT_THRESHOLD`.
    colmap = Vector{Union{Nothing, Dict{Int, Int}}}(nothing, n)
    @inbounds for m in 1:n
        r = SparseArrays.nzrange(L, m)
        if length(r) > _SELECTED_INVERSE_DICT_THRESHOLD
            d = Dict{Int, Int}()
            sizehint!(d, length(r))
            for idx in r
                d[rows[idx]] = idx
            end
            colmap[m] = d
        end
    end

    # Position of Σ[row, col] (row >= col) in `Σnz`/`rows`. `row` is
    # guaranteed structurally present in column `col` by the clique
    # argument in the docstring above.
    _pos(col, row) = let d = colmap[col]
        if d === nothing
            a, b = colptr[col], colptr[col + 1] - 1
            @inbounds while a < b
                mid = (a + b) >>> 1
                rows[mid] < row ? (a = mid + 1) : (b = mid)
            end
            a
        else
            d[row]
        end
    end

    @inbounds for j in n:-1:1
        r = SparseArrays.nzrange(L, j)
        lo, hi = first(r), last(r)
        Ljj = vals[lo] # rows[lo] == j: CSC stores rows sorted ascending, and j is the smallest row in a lower-triangular column.
        for idx in (lo + 1):hi
            i = rows[idx]
            s = zero(T)
            for idx2 in (lo + 1):hi
                k = rows[idx2]
                # max/min: Σ only stores i >= j positions.
                pos = k >= i ? _pos(i, k) : _pos(k, i)
                s += vals[idx2] * Σnz[pos]
            end
            Σnz[idx] = -s / Ljj
        end
        sd = zero(T)
        for idx2 in (lo + 1):hi
            # Σ[rows[idx2], j] was just written above at this same nzval
            # position -- idx2 addresses it directly, no lookup needed.
            sd += vals[idx2] * Σnz[idx2]
        end
        Σnz[lo] = 1 / Ljj^2 - sd / Ljj
    end
    return Σ
end

"""
    _depermute_selected_inverse(Σp::SparseArrays.SparseMatrixCSC{T}, p::AbstractVector{<:Integer}, n::Int) where {T} -> SparseArrays.SparseMatrixCSC{T, Int}

Internal step of [`selected_inverse`](@ref): map the lower-triangular,
permuted-order selected inverse `Σp` (as returned by
[`_selected_inverse_permuted`](@ref)) back to the original row/column
ordering implied by the fill-reducing permutation `p` (`p[i]` is the
original index of permuted row/column `i`, i.e. the matrix `Σp` inverts is
`A[p, p]`). Both triangles are materialized in the result, since a
permutation does not preserve the lower/upper distinction.

# Returns
- `SparseArrays.SparseMatrixCSC{T, Int}`, symmetric, size `(n, n)`.
"""
function _depermute_selected_inverse(Σp::SparseArrays.SparseMatrixCSC{T}, p::AbstractVector{<:Integer}, n::Int) where {T}
    nz = SparseArrays.nnz(Σp)
    I = Vector{Int}(undef, 2nz)
    J = Vector{Int}(undef, 2nz)
    V = Vector{T}(undef, 2nz)
    c = 0
    rows = Σp.rowval
    vals = Σp.nzval
    @inbounds for j in 1:n
        for idx in SparseArrays.nzrange(Σp, j)
            i = rows[idx]
            v = vals[idx]
            a, b = p[i], p[j]
            c += 1
            I[c] = a
            J[c] = b
            V[c] = v
            if a != b
                c += 1
                I[c] = b
                J[c] = a
                V[c] = v
            end
        end
    end
    resize!(I, c)
    resize!(J, c)
    resize!(V, c)
    return SparseArrays.sparse(I, J, V, n, n)
end

"""
    selected_inverse(H::SparseArrays.SparseMatrixCSC{T}; shift::Real = zero(T)) where {T} -> SparseArrays.SparseMatrixCSC{T, Int}

Compute exact entries of `inv(H + shift * I)` at every position covered by
the fill-in pattern of `H`'s sparse Cholesky factorization -- a superset of
`H`'s own nonzero pattern -- via Takahashi's selected-inversion recursion
([`_selected_inverse_permuted`](@ref), then
[`_depermute_selected_inverse`](@ref)).

`H` must be symmetric positive definite with only its lower triangle stored
explicitly (matching the storage convention of `CholeskySolverCache.H` in
`psf_photometry_simultaneous.jl`); the upper triangle is inferred by
symmetry, exactly as `Symmetric(H, :L)` does for the Cholesky factorization
itself.

Unlike a naive `Symmetric(H, :L) \\ I`, this never forms a dense `n x n`
inverse: after one sparse Cholesky factorization `H[p, p] = L * L'`, the
recursion recovers entries of `inv(H)` at the same asymptotic cost as the
factorization itself (dominated by fill-in, not by `n^3`), restricted to the
positions where the Cholesky factor `L` has a structural nonzero. That
restriction is not a limitation for block-structured normal-equations
matrices such as the per-star Hessian in `fit_all_stars_simultaneous`: every
entry of `H`'s own nonzero pattern is, by construction, also a nonzero of
`L` (fill-in only adds entries during elimination, never removes them), so
every diagonal block and every directly-coupled off-diagonal block of `H`
is recovered exactly. See [`selected_inverse_diagonal_blocks`](@ref) for
the common case of extracting per-block marginal covariances.

# Arguments
- `H::SparseArrays.SparseMatrixCSC{T}`: symmetric positive definite matrix
  with only the lower triangle stored.

# Keyword arguments
- `shift::Real = zero(T)`: diagonal shift applied before factorization
  (`H + shift * I`), e.g. for conditioning a near-singular block.

# Returns
- `SparseArrays.SparseMatrixCSC{T, Int}`: a symmetric sparse matrix (both
  triangles stored) in `H`'s original row/column ordering, holding exact
  entries of `inv(H + shift * I)` at the fill-in pattern and `0` elsewhere.

!!! note
    Querying a position outside the fill-in pattern silently returns `0`,
    not the true (generally nonzero, but not recovered) inverse entry
    there. Callers must restrict themselves to positions known to lie
    within `H`'s own nonzero pattern.

# Examples
```jldoctest
julia> using CrowdPhot, SparseArrays, LinearAlgebra

julia> H = sparse(LowerTriangular([4.0 0.0; 1.0 4.0])); # dense 2x2: fully within the fill-in pattern

julia> Σ = CrowdPhot.selected_inverse(H);

julia> Σ ≈ inv(Matrix(Symmetric(H, :L)))
true
```
"""
function selected_inverse(H::SparseArrays.SparseMatrixCSC{T}; shift::Real = zero(T)) where {T}
    n = size(H, 1)
    size(H, 2) == n || throw(ArgumentError("H must be square, got size $(size(H))"))
    F = cholesky(Symmetric(H, :L); shift = T(shift))
    p = F.p
    L = SparseArrays.sparse(F.L)
    Σp = _selected_inverse_permuted(L)
    return _depermute_selected_inverse(Σp, p, n)
end

"""
    selected_inverse_diagonal_blocks(H::SparseArrays.SparseMatrixCSC{T}, blocksize::Int; shift::Real = zero(T)) where {T} -> Array{T, 3}

Convenience wrapper around [`selected_inverse`](@ref) for a block-structured
normal-equations matrix `H` with `n = blocksize * nblocks` rows: return
every `blocksize x blocksize` diagonal block of `inv(H + shift * I)` -- the
exact marginal covariance of each block's parameters, including coupling
through `H` with every other block, not just the block's own diagonal --
stacked as `[:, :, b]` for block `b = 1:nblocks`. This matches the
`(p, p, n_active)` layout of `Hdiag` in `psf_photometry_simultaneous.jl`.

# Arguments
- `H::SparseArrays.SparseMatrixCSC{T}`: symmetric positive definite, lower
  triangle stored; `size(H, 1)` must be a multiple of `blocksize`.
- `blocksize::Int`: number of parameters per block (e.g. `3` for `(y, x,
  flux)`).

# Keyword arguments
- `shift::Real = zero(T)`: forwarded to [`selected_inverse`](@ref).

# Returns
- `Array{T, 3}` of size `(blocksize, blocksize, nblocks)`.

!!! note
    Every diagonal block lies within the fill-in pattern recovered by
    [`selected_inverse`](@ref) (it is part of `H`'s own nonzero pattern, a
    subset of the fill-in pattern), so this never silently returns zeros
    for a diagonal-block entry.
"""
function selected_inverse_diagonal_blocks(H::SparseArrays.SparseMatrixCSC{T}, blocksize::Int; shift::Real = zero(T)) where {T}
    n = size(H, 1)
    nblocks, r = divrem(n, blocksize)
    r == 0 || throw(ArgumentError("size(H, 1) = $n is not a multiple of blocksize = $blocksize"))
    Σ = selected_inverse(H; shift)
    out = zeros(T, blocksize, blocksize, nblocks)
    @inbounds for b in 1:nblocks
        base = (b - 1) * blocksize
        for l in 1:blocksize, k in 1:blocksize
            out[k, l, b] = Σ[base + k, base + l]
        end
    end
    return out
end
