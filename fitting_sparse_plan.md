# Simultaneous ("sparse") PSF Fitting Plan

Branch target: `sparse-matrix`.

Goal: add a second whole-image fitting algorithm that optimizes the parameters
of every source at once, swappable for the existing DOLPHOT-style sequential
`fit_all_stars` with every other pipeline stage (background, detection,
morphology, PSF model) held fixed.

Reference implementations: crowdsource builds an explicit sparse `J` and solves
it with LSQR over flux-only columns; psf_sandbox runs a damped Gauss-Newton /
LM loop over `3N` parameters matrix-free over `J`. Neither is a spec for this
code; the Julia measurements that decide the design are recorded in
`experiments/sparse-fitting/RESULTS.md` (not in this document). This plan keeps
only the numbers that bind a decision.

## 1. Design decisions, stated up front

1. **The solve object is the normal-equation matrix `H = J'WJ`, never `J`.**
   Measured 12-23x faster per Krylov iteration than any `J`-based method, with
   a better captured objective per iteration, because `nnz(H)` is 7-22x smaller
   than `nnz(J)` at these densities. The reference codes' "never form the
   normal equations" was correct against `scipy.sparse`; it is wrong in Julia.
2. **`H` is accumulated directly from stamps, never as `J' * J`.** `J'*J`
   churns 79 GB of allocation at `N = 5e5` and costs more than the rest of the
   iteration; direct accumulation is exact (0.0 max diff) and allocates 5-9x
   less.
3. **Default solver is preconditioned CG on `H`; Cholesky is the exact oracle.**
   CHOLMOD on `H` is fast at `N ≤ 2e5` (130 ms/retry at `N = 1e5`) but costs
   41.9 s/retry and 24.5 GB peak at `N = 5e5`, `L = 4000`. `:cg` is the only
   viable choice at the target.
4. **Jacobian values live in a stamp array (no `J` matrix), and the stamp
   anchor is fixed for the whole fit.** This makes the pixel footprint, the
   `H` sparsity pattern, and the neighbor list constant across the fit.
5. **No lookup table for PSF rendering.** Direct evaluation, filled once per
   linearization. LUTs buy ~2.9x on the render after accounting for the
   vectorization gap, and their systematic error (8.8e-4 at 17x17 phase)
   exceeds photon noise above SNR ~1100 — exactly the wrong artifact for an
   accuracy benchmark.
6. **No robust reweighting (IRLS) in the simultaneous path.** See §4.5.
7. **No multi-band, no per-star shape parameters, no detection/merge/split,
   no global sky basis.** See §9 and §10.

## 2. Goals and non-goals

Goals:

- `fit_all_stars_simultaneous` returns a `MultiPassPhotResult` interpretable
  field-for-field against `fit_all_stars`.
- Every non-fitting concern (background, detection, morphology, PSF model,
  diagnostics math, error extraction, convergence tests) is shared code, not a
  parallel implementation.
- A separate public function; `fit_all_stars` is untouched.
- Honest comparison: same tolerances, same weighting, same stamp pixels (to
  the one-pixel boundary difference documented below).

Non-goals for v1:

- Multi-band / multi-epoch (crowdsource's `B` axis).
- Per-star shape parameters (`fwhm`, `y_fwhm`, `x_fwhm`, `theta`, ...) as free
  variables. Error explicitly; see §4.1.
- Source detection, merging, splitting, or culling inside the fit.
- A global sky basis (crowdsource's `nskyx`/`nskyy` spline columns).
- Correct marginal (neighbor-aware) errors via selected inversion.
- GPU.

## 3. Formulation

Let star `i` have free parameter vector `θ_i ∈ R^p`, `p = length(free_idx)`,
in CrowdPhot field order. For the models this fitter supports (see §4.1),
the block is exactly `(y, x, flux)`, so `p = 3`. Global parameter vector
`θ ∈ R^{3N}` with parameter `k` of star `i` at global index `3(i-1) + k`.

Conventions match `lm_irls` / `psf_fitting.jl` exactly:

- residual `r_q = M(θ)_q - data_q` (model minus data),
- weighted residual `r̃_q = sqrt(w_q) r_q`,
- Jacobian `J[q, 3(i-1)+k] = sqrt(w_q) * ∂m_i/∂θ_{i,k}[q]`,
- gradient `b = J' W r` (note the `+` sign),
- normal matrix `A = J' W J`,
- step `δ = -A_damp⁻¹ b`,
- predicted reduction `prered = -2 δ'b - δ'Aδ`.

`M(θ) = Σ_i m_i(θ_i)`, cost `C(θ) = Σ_q w_q r_q²`. Each Gauss-Newton step
solves

```
min_δ ||Jδ + r̃||² + λ ||D δ||²
```

in **parameter coordinates**, where `D = diag(colnorm)` is the column-norm
diagonal. Column equilibration (§4) transforms this to a solver in
**equilibrated coordinates**:

```
δ_scaled = colnorm ⊙ δ,    H_scaled = D⁻¹ J'WJ D⁻¹,    b_scaled = D⁻¹ b,
(H_scaled + λ I) δ_scaled = -b_scaled
```

`diag(H_scaled) = 1` exactly, so the Marquardt shift `λ · max(diag, 1e-6)`
reduces to the plain `λ I` shift shown above. This is the same single `λ`
as `lm_irls` (`damp!`, `src/levenberg_marquardt.jl:74-79`); **do not square
it.** Only `:lsqr`/`:lsmr` (reference paths) square `λ` internally, and they
must be passed `sqrt(λ)`.

The model is linear in flux and nonlinear only in position, so a Gauss-Newton
step is exact in the flux directions; the outer iteration is driven by the
position nonlinearity.

## 4. The comparability contract

### 4.1 Free parameters are exactly a subset of `(y, x, flux)`

`fit_all_stars` fits `bkg` per star by default. In a simultaneous solve,
`N` nearly-constant columns over overlapping stamps is a degenerate system
(a constant offset under one star's stamp is almost reproducible by adjusting
neighbor fluxes). Neither reference code solves for per-star sky. Same
conclusion as `dolphot_crowding_fixed_vs_free_sky.md`.

Contract: the comparison runs **both** arms with the background
pre-subtracted (`Background2D`) and `fixed = (; bkg = zero(FT))`. The
simultaneous path errors, not warns, if its free set is not a subset of
`(:y, :x, :flux)`:

```julia
offenders = setdiff(free_names, (:y, :x, :flux))
isempty(offenders) || throw(ArgumentError(
    "fit_all_stars_simultaneous fits only (y, x, flux) per star; got free " *
    "parameters $(offenders). Fix `bkg` (subtract the background first and " *
    "pass `fixed = (; bkg = zero(eltype(image)))`), and fix all shape " *
    "parameters. Pass the same `fixed` to `fit_all_stars` for a " *
    "like-for-like comparison."))
```

This subsumes the per-star shape-parameter exclusion: `CircularGaussianPSF`
with `fixed = (; bkg)` would otherwise silently free `fwhm`.

### 4.2 Same pixels, up to the one-pixel boundary difference

`fit_all_stars` builds its box as `floor(Int, y - fit_rad):ceil(Int, y +
fit_rad)` clamped by `_clamp_inds`, recomputed from the current fractional
position (`psf_photometry_single.jl:330-334`). The simultaneous solver needs a
fixed-size stamp:

- Solver stamp: half-width `R = round(Int, fit_rad) + 1` (one extra ring), so
  `S = 2R + 1`, anchored once on the rounded **initial** position and held
  fixed for the whole fit. The extra ring keeps a star inside its stamp as it
  drifts (backstopped by `max_step`, §5.3, and the final validity gate).
  Fixed anchoring means the pixel set, the `H` pattern, and the neighbor list
  are built once (§5.2).
- Diagnostics: use the identical `floor`/`ceil` box rule as the sequential
  path, so `qfit`, `chisq`, `crowding` are computed over exactly the same
  pixels in both arms. Only the solver's internal footprint differs.

Document the one-pixel solver-footprint difference in the docstring.

### 4.3 Same weights

`inv_var` semantics unchanged: non-positive or non-finite means masked. Trap
the references recorded: zeroing the weight is **not** enough for a NaN data
pixel (`0 * NaN = NaN`, one such pixel makes the global cost NaN and rejects
every step while reporting success). Non-finite data pixels are zeroed AND
zero-weighted. Because `inv_var` may be `nothing`, the simultaneous path always
materializes an internal weight array over stamp pixels: `w = 1` everywhere
when `inv_var === nothing`, else `w = inv_var[pix]` for finite-positive values
and `0` otherwise, and error up front if `inv_var === nothing` and the image
is not all-finite.

### 4.4 Same convergence tests, with two stated deviations

Do not refactor `lm_irls`; duplicate the expressions in the new file, each
commented with the `levenberg_marquardt.jl` line it came from. `lm_irls` is
the control. The quantities, in **parameter coordinates**:

| `lm_irls` quantity | simultaneous equivalent | source |
| --- | --- | --- |
| `b = J'Wr` | `b_true = colnorm ⊙ b_scaled` | `levenberg_marquardt.jl:278` |
| `A_ii` | `colnorm_i²` | computed at stamp fill |
| `dot(δ, A, δ)` | `dot(δ_scaled, H_scaled, δ_scaled)` | `levenberg_marquardt.jl:564` |
| `prered` | `-2 δ_scaled'b_scaled - δ_scaled'H_scaled δ_scaled` | `levenberg_marquardt.jl:564` |

- `g_converged`: `max_i |b_true_i| / (sqrt(A_ii * C) + eps) <= g_tol`
  (`levenberg_marquardt.jl:535-540`).
- `f_converged`: MINPACK three-part `prered` test
  (`levenberg_marquardt.jl:578-581`).
- `x_converged`: `norm(D .* δ) <= x_tol * (norm(D .* x) + x_tol)`
  (`levenberg_marquardt.jl:584`), with the running maximum
  `D[i] = max(D[i], colnorm[i])` per accepted step
  (`levenberg_marquardt.jl:603`).

**Deviation 1 — `D` is a running maximum, not the current `colnorm`.**
After equilibration `D ≠ colnorm` once any step is accepted, so
`norm(D .* δ) ≠ norm(δ_scaled)`. Carry `D` separately and compute the
`x_tol` test in parameter coordinates as `norm((D ./ colnorm) .* δ_scaled)`.
Do not silently substitute `norm(δ_scaled)`; that changes `x_converged`.

**Deviation 2 — the denominator is the reduced cost, `C_r = cost / dof`.**
`lm_irls` divides `g`/`f` tests by the *absolute* cost; per star that is
`~S²`, globally it is `~npix`, so the same `g_tol` means `~sqrt(S²/npix)` as
strict globally (≈`3e-2` per star at `L = 4000`) and, worse, the effective
tolerance depends on `N` and `L`. That would make the §7 sweep measure the
stopping rule rather than the algorithm. Use `C_r = cost / (N_union_pix -
3*N_active)` in both the `g` and `f` denominators and document the deviation
next to the duplicated expressions. Also report the per-star maximum scaled
gradient to the trace (not the return value) so convergence quality is
auditable.

The sign convention above (`b = +J'Wr`, `δ = -A⁻¹b`, `prered = -2δ'b - δ'Aδ`)
is exactly the code's; use it verbatim in every formula so the duplicated
tests transfer literally.

### 4.5 No robust reweighting

`fit_all_stars` in practice runs with an `inv_var` array, which orients
`lm_irls` to fixed weights and never activates IRLS. The simultaneous path
does not support it: `reweight`, `scale_estimator`, `weight_reset_tol` are not
accepted keywords. A naive port would also be wrong: `estimate_scale` on one
global residual vector (dominated by sky) returns a scale much smaller than
the per-star scale in a crowded field, so the two arms would diverge for
reasons unrelated to the algorithm. If wanted later: per-star scale over each
stamp, overlapping pixels take the minimum robust weight (the only
order-independent choice).

### 4.6 Errors are apples-to-apples

`fit_all_stars` takes `cov` from the per-star LM normal matrix against a
neighbor-subtracted image; the simultaneous solver inverts the per-star
diagonal block of `H`. Both ignore covariance with blended neighbors, so
`flux_err`/`y_err`/`x_err` compare directly. Neither is a correct marginal
error in a crowded field (the per-star block is ~2x optimistic); say so in
the docstring. The simultaneous formulation is the only one that could
produce the correct marginal via selected inversion — deferred.

## 5. Core data structures

### 5.1 `StampDerivatives` (the stamp operator)

The Jacobian values are stored per stamp; `J` is never materialized. The
default solver iterates on `H`, not `J`, so this struct's hot-path duties are
`apply_JT!` (to form `b`), `colnorm`, and supplying blocks to the `H`
accumulator. `apply_J!` exists only for the reference `:lsqr`/`:lsmr` paths
and the adjoint test.

```julia
struct StampDerivatives{T, I <: Integer}
    values::Array{T, 3}   # (p, S², n_active); weighted, then column-equilibrated (raw ./ colnorm)
    pixels::Matrix{I}      # (S², n_active); flat pixel index, 0 = masked/off-image
    colnorm::Matrix{T}    # (p, n_active); the true per-column norm, kept from fill time
    npix::Int
    p::Int
end
```

- `apply_JT!(z, Jm, u)`: `z[3(i-1)+k] = Σ_m values[k, m, i] * u[pixels[m, i]]`.
- `apply_J!(y, Jm, v)`: `y[pixels[m, i]] += Σ_k values[k, m, i] * v[3(i-1)+k]`
  (zero `y` first). Reference paths only.
- **Equilibration, computed at fill.** For each star and parameter, compute the
  raw weighted derivative per pixel, reduce
  `colnorm[k, i] = sqrt(Σ_m raw[k, m, i]²)` over the stamp, then store
  `values[k, m, i] = raw[k, m, i] / colnorm[k, i]` (floor `colnorm` at `eps(T)`).
  With column-scaled `values`, `apply_JT!(r)` *is* `b_scaled = D⁻¹J'r` directly
  (since `(JD⁻¹)' = D⁻¹J'`), and `diag(H_scaled) = 1` by construction. `colnorm`
  is the true per-column norm kept at fill time — it is NOT recoverable from
  `diag(H_scaled)` (which is exactly 1).
- Sentinel `0` for off-image and zero-weight pixels. Handle branchlessly:
  `valid = r != 0; rr = ifelse(valid, r, 1); uu = ifelse(valid, u[rr], zero(T))`
  (load from a safe index, mask the value). `@turbo` rejects a raw `if`.
- Thread `apply_JT!` (trivially thread-safe over stars); leave `apply_J!` as a
  plain `@inbounds` loop (stamps overlap; per-thread scratch measured 4.8x
  slower than serial).

The render into `values` should mirror the per-model `_accum_*!` kernels in
`src/psf/psf_fitting.jl` (`_accum_circular_gaussian!`, `_accum_image_psf!`,
`_accum_gridded_imagepsf!`, ...), which are the fast path `fit_star` actually
uses — not a generic `evaluate_fg` loop. Those kernels already compute value +
`(dy, dx, dflux)` into scalars and write them out, which is the shape a
vectorized stamp fill needs. Measure the fill against them as the throughput
ceiling.

Vectorization of the value+gradient fill, settled by
`experiments/simd/05_evaluate_fg.jl`: `evaluate_fg`'s `(f, G::SVector)`
return cannot be consumed correctly inside a `@turbo` loop — indexing `G[k]`
or converting with `Tuple(G)` both fail (and the `G[k]` case can silently read
a stale global of the same name). The pattern that *does* vectorize is a flat
multi-value return destructured positionally at the callsite:

    f, gy, gx, gfwhm, gflux, gbkg = evaluate_fg_flat(model, y, x)

with `LV.can_turbo(::typeof(evaluate_fg_flat), ::Val{3}) = true` — measured
~14x over a scalar loop at 32x33, ~6x at 5x5. The distinguishing factor is not
`SVector` vs `Tuple`: a nested `(f, container)` that is later indexed breaks
`@turbo`; a flat multi-value return destructured straight into scalars works.

Arity differs across model types, so a single generic `evaluate_fg` cannot
serve one generic destructuring loop — but §4.1 already restricts the free set
to `(y, x, flux)`, so the stamp fill only ever needs the *uniform* four-value
tuple `(f, df_dy, df_dx, df_dflux)` for every model. Recommend one internal
helper

    _evaluate_fg_fluxpos(model, y, x) -> (f, df_dy, df_dx, df_dflux)

with a per-model method for each PSF type, `LV.can_turbo(_evaluate_fg_fluxpos,
Val{3}) = true`, and one generic `@turbo` fill loop that destructures the four
values positionally — mirroring how the generic value-only `evaluate` already
vectorizes via a single `can_turbo` on the generic. Keep the derivative order
matching the existing field order (y before x) so the `values` planes line up
with `free_idx` slicing and error extraction. Validate the generic-flat
combination with a small experiment first (`05_evaluate_fg.jl` tested a
*concrete* flat function, not a generic one with per-model methods); if generic
dispatch does not vectorize, fall back to per-model `@turbo` fill kernels.
`evaluate_fg` itself stays untouched as the scalar reference.

### 5.2 `H` and the neighbor model

`H` is a symmetric `SparseMatrixCSC` over `3*N_active` unknowns, stored as one
triangle under `Symmetric` (halves both memory and the CG matvec). Nonzero
blocks:

- diagonal `3×3` block `H_ii` for every star;
- off-diagonal `3×3` block `H_ij` iff star `i` and `j` share a pixel.

**The neighbor list is the sparsity pattern of `H`**, and building it is the
same work as filling `H`, not an extra cost: `Σ_pixels k_p² = Σ_pairs
|overlap|`. Because the anchor is fixed (§4.2), the pattern is built **once**
before the outer loop and never changes.

Build once:

1. Pixel→star bucket: for each active star, push its index into a bucket per
   covered pixel (one pass over `pixels`, `O(S²N)`).
2. Emit unordered pairs `(i, j)` for each pixel with `k ≥ 2` covering stars;
   dedupe into a sorted neighbor list per star. Total pairs
   `= nnz(H)/p²` blocks (minus diagonals).
3. Precompute, per pair, the shared-pixel list (the overlap footprint) once —
   it is constant for the whole fit.
4. Allocate `H`: `colptr`/`rowval` from the neighbor lists (3 columns per
   star), `nzval` filled by the accumulator.

`H` **values** are refilled every outer iteration (positions move), pattern
never. Store the lower triangle and wrap as `Symmetric(H, :L)`: the CG matvec
reads through that wrapper, and the `:cholesky` oracle reuses the symbolic
factorization via `cholesky!(F, Symmetric(H, :L); shift = λ)` on the unchanged
pattern.

Sizing (for memory planning; `density = N/L²`, `nbrs = 0.85-0.95 * density *
(2S)²`): `nnz(H) = N_active * (nbrs + 1) * p²`, in one triangle.

### 5.3 `H` accumulation (per outer iteration)

For each star `i` (block row `i`): gather the stamp derivative rows
`values[:, m, i]` for `m` in its pixel list. Zero a `p×p` scratch; for each
of `i`'s neighbors `j`, zero the block scratch and loop the pair's precomputed
shared pixels accumulating `Σ_m values[:, m_i, i] values[:, m_j, j]'`; write
into `H`. The diagonal block `H_ii` is the same loop with `j = i` (all `S²`
pixels). `colnorm` was already computed at fill (§5.1); `diag(H_scaled) = 1`
by construction, so do not re-derive it from `H`. Working type `Float64`.

Memory at `N = 5e5`, `r = 6`, `L = 4000`: `values` ~2.0 GB + `pixels`
(Int32) ~0.34 GB + `H` ~1.5 GB full / ~0.75 GB one triangle + `O(npix)`
images. Comfortable for `:cg`.

### 5.4 Damping and preconditioner

- Shift: `H_scaled + λ I` (equilibrated), passed into the operator — no matrix
  copy.
- **Preconditioner: block-Jacobi over each star's `3×3` diagonal block**,
  `M_i = (H_scaled_ii + λ I)⁻¹`. This is not the identity after equilibration
  (the off-diagonal y/x/flux coupling survives), and it is free: §5.5 already
  forms and inverts exactly these blocks. Plain Jacobi would be the identity
  (`diag(H_scaled) = 1`) and is pointless.

### 5.5 Errors

After convergence, rebuild the stamp values at the final `θ`. The per-star
block `H_ii = Σ_m values[:, m, i] values[:, m, i]'` is the diagonal block of
`H`. Rescale out of equilibrated units by `colnorm_i colnorm_i'`, invert with
a Tikhonov floor (`H_ii + 1e-12 * tr(H_ii) * I`), take `sqrt` of the diagonal,
and route through the existing `_extract_errors!`
(`psf_photometry_single.jl:191`) so fixed parameters get `zero(T)` identically
to the sequential path. Honor `covariance_estimator` as `lm_irls` does
(`levenberg_marquardt.jl:366-388`).

## 6. The outer loop

One damped Gauss-Newton / LM loop, structurally parallel to `lm_irls` but
iterating on `H`, with the IRLS branch removed and a nested damping-retry loop
(as psf_sandbox, and unlike `lm_irls`'s flat one-step-per-iteration loop —
note the divergence explicitly).

```
active, pixels, neighbor list, H pattern  ← build ONCE from initial θ (fixed anchor)
θ ← initial catalog; C ← cost(θ)
D ← colnorm(θ)          # initial scaling; running max thereafter
for iter in 1:max_iter
    values, colnorm ← render value+gradient at θ  # colnorm at fill; values equilibrated (§5.1)
    H_scaled ← accumulate from stamps (§5.3)     # one triangle; diag == 1
    b_scaled ← apply_JT!(r)                      # == D⁻¹J'r already (§5.1)
    g_converged check (b_true = colnorm .* b_scaled; reduced-cost denominator) && break
    for trial in 1:max_trials
        δ_scaled ← pcg(H_scaled + λI, -b_scaled; M, inner_iterations)
        δ ← δ_scaled / colnorm
        cap_position_step!(δ, max_step)        # §6.3 (y, x sub-block only)
        δ_scaled ← colnorm .* δ                # re-sync: prered must use the capped step
        M_cand ← render_model(θ + δ)           # value-only, fixed stamps
        C_cand ← Σ w (M_cand - data)²
        prered ← -2 δ_scaled'b_scaled - δ_scaled'H_scaled δ_scaled
        ρ ← (C - C_cand) / prered
        accept / adjust λ (λ_up/λ_down/λ_min/λ_max as lm_irls)
    end
    # after an accepted step: θ, C ← candidate; D[i] = max(D[i], colnorm[i])
    x_converged / f_converged checks && break
end
```

Three non-textbook choices, all carried from the references' recorded
experience:

### 6.1 Truncate the inner solve hard

Solving a stale linearization precisely is wasted past a point. Default
`inner_iterations = 5`. Expose it.

### 6.2 Damping is a diagonal shift on `H`

`H_scaled + λ I` in equilibrated coordinates (see §3; single `λ`, Marquardt).
For `:cg` pass the shift into the operator; for `:cholesky` use `cholesky!(F,
H; shift = λ)`. `H` is accumulated once per linearization and reused across
every damping trial. `MarquardtDamping` is the only supported damping;
`LevenbergDamping` (equilibration off) is rejected with an informative error
because it conflicts with the `x_tol` scaling.

### 6.3 Cap the position step in pixels, per star

The position step is capped in pixels, not sigma: PSF linearization fails at a
physical scale (~FWHM), and a damping-based limit scales as `σ/λ`, pinning
bright stars too tight while leaving faint ones free. Scale only the `(y, x)`
sub-block of each star's `δ` (leave flux), and do it per star (a global line
search would let one runaway star throttle the rest). Default `max_step =
fit_rad`, which is self-contained (no PSF width accessor needed) and errs
generous, the mild failure direction. A FWHM-based refinement is a follow-on.

### 6.4 Cost evaluation per trial

Each damping trial moves `θ`, so `cost(θ+δ)` must be re-evaluated. It is a
**value-only** render over the fixed stamps (no gradients, no `H`), followed by
a residual-and-cost pass over the union of stamps — the cheap half of a
linearization. On acceptance, the candidate residual becomes the current
residual; on rejection, `λ` increases and the same `H_scaled`, `b_scaled` are
reused. `H` and `b` stay frozen across retries (stale linearization, fresh
candidate model), matching psf_sandbox.

## 7. Validation gate

`free_names ⊆ (:y, :x, :flux)` is enforced once (§4.1). The `active` star list
(see failure policy) is built once, before the outer loop, and held fixed so
`x_converged` and the trust-region state always refer to one problem.

## 8. Diagnostics

`MultiPassPhotResult` carries `chisq`, `qfit`, `qfit_expected`, `qfit_z`,
`crowding`, computed in `fit_all_stars` in one inline block
(`psf_photometry_single.jl:372-432`). Extract that math into
`_star_diagnostics!(...)` in a new shared file and call it from **both** arms
(the existing `test/photometry/psf_photometry_single.jl` suite pins the
sequential behavior, so the baseline risk is verifiable and small). The
simultaneous path feeds it directly:

- `qfit = Σ|global_residual| / flux` over the box — the global residual
  `data - M(θ)` plus star `i`'s own model is exactly the sequential path's
  neighbor-subtracted residual.
- `chisq = Σ w r² / (n_pix - p)` over the box, same global residual.
- `crowding` uses `F_clean` from `global_residual + own model` and `F_dirty`
  from the original `image`, with the same PSF-weighted linear flux estimator.
- `qfit_expected`/`qfit_z` depend only on `inv_var` and the box; unchanged.

The convergence expressions of §4.4 are duplicated (not shared) in the
simultaneous file, each citing its `levenberg_marquardt.jl` line; only the
diagnostics block is extracted.

## 9. Failure policy

The sequential algorithm drops stars via `valid` and survives per-star
exceptions in a `try`/`catch`; a shared system has no equivalent — one
degenerate star poisons the solve. Policy:

| Condition | Detection | Result |
| --- | --- | --- |
| Stamp touches image edge | any sentinel in `pixels[:, i]` | keep and fit |
| No usable pixels | `all(iszero, pixels[:, i])` or zero `colnorm` | `valid[i] = false`; excluded from `active` before the solve |
| Flux non-positive | `flux[i] <= 0` after convergence | `valid[i] = false` |
| Non-finite parameters | `!isfinite` | `valid[i] = false` |

Zero-weight stars are excluded from the parameter vector (`active::Vector{Int}`
index map), not merely flagged: a zero column is a null direction LSQR will
wander in and makes `colnorm` zero. `active` is built once and held fixed; a
star that drifts off-image or into a mask mid-fit is caught by the final
`isfinite` / `flux > 0` gate, the same place the sequential path catches it.
`n_stars == 0` or `isempty(active)` returns the same empty
`MultiPassPhotResult` as `fit_all_stars` (`psf_photometry_single.jl:266-269`).

`n_failed`/`failure_msgs` keep their meaning (stars excluded before the
solve). Per the current scope, the return fields carry simplified semantics:
`converged .= true` for all stars, `n_passes` = the number of outer LM
iterations actually run, `n_iter .= n_passes` for every star. No solver
diagnostics (λ history, CG counts, per-iteration cost) are returned; they
belong in the trace/experiment scripts if needed. Document the meaning shift
of `n_passes`/`n_iter` in the docstring.

## 10. API

```julia
fit_all_stars_simultaneous(image, psf, sources, fit_rad;
                           fixed::NamedTuple = (;),
                           inv_var = nothing,
                           kws...) -> MultiPassPhotResult
```

`sources` is parsed by the same three `_extract_source_catalog` methods as
`fit_all_stars` (`Vector{<:NamedTuple}`, `NamedTuple`, `MatchedFilterResult`).

A separate function, not an `alg::Symbol` keyword: two functions sidestep the
divergent-keyword problem (`n_passes` on one side, `inner_iterations` on the
other), and Julia's keyword handling produces the error for free. Export
alongside `fit_all_stars`. Unifying behind one entry point is a later
decision.

Keywords:

| Keyword | Default | Meaning |
| --- | --- | --- |
| `inner_iterations::Int` | `5` | CG iteration cap per trial step |
| `max_step::Real` | `fit_rad` | per-star position cap in pixels |
| `solver::Symbol` | `:cg` | `:cg` (block-Jacobi PCG on `H`), `:cholesky` (exact oracle) |
| `max_trials::Int` | `8` | damping retries per outer iteration |

Shared with `fit_star`/`lm_irls`, same names and meanings: `max_iter`,
`x_tol`, `f_tol`, `g_tol`, `λ_init`, `λ_up`, `λ_down`, `λ_min`, `λ_max`,
`damping`, `show_trace`, `covariance_estimator`. **Not accepted** (see §4.5):
`reweight`, `scale_estimator`, `weight_reset_tol`.

`max_iter` defaults to `40` (global outer iterations), not `fit_star`'s `200`
per star. Document why.

## 11. File layout and dependencies

```
src/photometry/psf_photometry_single.jl        # existing; changed only to call
                                               #   the shared _star_diagnostics!
src/photometry/psf_photometry_diagnostics.jl   # NEW: _star_diagnostics! (shared)
src/photometry/psf_photometry_simultaneous.jl  # NEW: StampDerivatives, apply_J!,
                                               #      apply_JT!, H accumulation,
                                               #      PCG, outer loop,
                                               #      fit_all_stars_simultaneous
test/photometry/psf_photometry_simultaneous.jl # NEW; add @safetestset in runtests.jl
```

`MultiPassPhotResult` stays in `psf_photometry_single.jl`; include the
simultaneous file after it.

Dependencies: add `SparseArrays` (stdlib) with `[compat] SparseArrays = "1"`.
Do not add `Krylov.jl`: the `:cg` path needs in-house preconditioned CG (~60
lines, no bidiagonalization subtlety); `:lsqr`/`:lsmr` are dropped as reference
paths. Do not add `LinearOperators.jl` (a dependency purchased solely to wrap
two functions we own).

## 12. Implementation sequence

Pre-check: the binding numbers quoted in §1 (the `:cg`-over-`J` decision, the
Cholesky scaling at `N = 5e5`, the `nnz(H)` sizing) come from
`experiments/sparse-fitting/RESULTS.md`, whose scripts (A, B) are not yet in
the tree. The gates below re-derive the correctness facts; the performance
facts only set defaults, so mark them provisional and re-measure in the §13
harness (or recreate A/B first) before trusting the `:cg` default at scale.

Build and gate in this order; each gate must pass before the next:

1. `StampDerivatives` + `apply_JT!`/`apply_J!` on a synthetic `randn` stamp
   set. Gate: adjoint identity `dot(u, J*v) == dot(J'u, v)`.
2. `H` accumulation from stamps. Gate: max-abs agreement against dense `J'WJ`
   at `N = 50` (exact to round-off) and against `J'*J` for one case.
3. Block-Jacobi PCG step. Gate: matches the dense damped solve `(H + λI)\b`
   to tight tolerance at `N = 50`.
4. Outer LM loop on one isolated star. Gate: matches `fit_star` in `y`, `x`,
   `flux`, and errors to fitting tolerance.
5. Errors (§5.5), diagnostics (§8), failure policy (§9).
6. `experiments/sparse-fitting/` comparison harness (§13).

## 13. Validation tests

In addition to the gates above:

1. Finite-difference check: `b` and `diag(H)` against the global cost (catches
   weight, sentinel, and equilibration errors the adjoint identity cannot).
2. Single isolated star vs `fit_star` (gate 4, kept as a test).
3. Recovery against truth: `simulate_sources` at several crowding levels; bias
   and scatter in `flux`, `y`, `x` for both algorithms; monotonic statistical
   assertions (scatter improves with SNR, crowding bias worsens with density),
   one `StableRNG` per testset.
4. Residual images: compare `result.residual` between the two arms at matched
   configuration; systematic structure in the difference is the most
   informative divergence diagnostic.
5. Failure paths: fully-masked star, star entirely off-image, star driven to
   negative flux, all-`NaN` region — each produces the documented `valid` state
   and a `failure_msgs` entry where excluded before the solve.

## 14. Comparison harness

Everything under `experiments/sparse-fitting/` (its own `Project.toml`,
numbered scripts, `run_all.jl`, `RESULTS.md`), following `experiments/simd/`.
Nothing in `benchmark/` until the comparison is run.

The core script runs both arms on identical `simulate_sources` input with
identical `fixed`, `inv_var`, `fit_rad`, and tolerances, and reports per
crowding level: wall time, allocations, flux/position bias and scatter vs
truth, reported-error vs actual-scatter ratio, `qfit`/`qfit_z`/`crowding`
distributions, and the invalid-star fraction.

### Sweep the iteration budget, not two defaults

A single point comparison of `n_passes = 3` vs `max_iter = 40` measures the
defaults. Sweep instead and report accuracy-vs-wall-time curves per crowding
level:

- `fit_all_stars`: `n_passes ∈ 1:5`.
- `fit_all_stars_simultaneous`: `max_iter ∈ {10, 20, 40, 80}`, and
  `inner_iterations ∈ {3, 5, 10, 20}` at fixed `max_iter`.

Accuracy is a single scalar per run (RMS position error, fractional flux
error). The crossover is then a property of the curves. Both arms are serial
in v1: `fit_all_stars` is inherently sequential (progressive subtraction), so
a serial simultaneous implementation is the fair comparison; the simultaneous
arm's threading headroom is real and will not show up in these numbers — say
so in `RESULTS.md` so it is not read as a deficiency.

### Projected cost at the target scale

At `L = 4000`, `N = 5e5`, `r = 6`, one outer iteration:

| component | projected |
| --- | --- |
| render value+gradient | ~10 s |
| accumulate `H` from stamps | ~12.6 s |
| candidate renders (value-only, ~1.5 trials) | ~7.5 s |
| CG, 5 iterations | ~0.6 s |
| **total** | **~30 s** |

~20 minutes for a 40-iteration fit, dominated by the render and the `H`
accumulation; the linear solve is a rounding error. If the harness measures
something far from this, chase the discrepancy first.
