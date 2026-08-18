using CrowdPhot: CrowdPhot, besselj1
using Bessels: Bessels
using Test

# Accuracy tests for the branchless `besselj1`/`_besselj1` port in
# src/bessels.jl, checked directly against Bessels.jl's own (branchy)
# reference implementation rather than reproducing its test suite. Points
# are chosen to cross every region boundary in the branchless port:
#   Float32: x <= 2 (rational near-origin form) vs. x > 2 (asymptotic form).
#   Float64: x <= pi/2 (low-order polynomial), pi/2 < x <= 26 (per-interval
#     table lookup, spanning several different intervals), 26 < x < 120 and
#     x >= 120 (asymptotic form, two rational-approximation orders).

@testset "besselj1" begin
    @testset "Float32" begin
        xs = Float32[0.0, 1.0f-6, 0.5, 1.0, 1.9, 2.0, 2.1, 5.0, 20.0, 100.0, 1.0f4, 1.0f6]
        xs = vcat(xs, -xs)
        for x in xs
            got = besselj1(x)
            @test got isa Float32
            @test got ≈ Bessels.besselj1(x) rtol = 1.0f-6 atol = 1.0f-7
        end
    end

    @testset "Float64" begin
        xs = Float64[0.0, 1.0e-8, 0.5, pi / 2, 1.0, 5.0, 13.0, 26.0, 50.0, 120.0, 500.0, 1.0e4]
        xs = vcat(xs, -xs)
        for x in xs
            got = besselj1(x)
            @test got isa Float64
            @test got ≈ Bessels.besselj1(x) rtol = 1.0e-10 atol = 1.0e-12
        end

        # Beyond ~1e6, the vendored port uses plain `cos(x_safe + xn)`
        # instead of upstream's double-double argument reduction (see
        # src/bessels.jl); accuracy degrades gracefully but is intentionally
        # not held to the same tolerance as the rest of the domain -- this
        # package's actual argument range (u = pi*r/a for pixel-scale r)
        # never approaches it.
        @test besselj1(1.0e8) ≈ Bessels.besselj1(1.0e8) rtol = 1.0e-6

        # x == Inf lands in the asymptotic branch (x > 26) without being
        # substituted away by the "keep inv/sqrt safe" x_safe guard, so it
        # exercises a separate finiteness guard on cos's argument -- see
        # src/bessels.jl for why this needs its own handling.
        @test besselj1(Inf) == 0.0
        @test besselj1(-Inf) == 0.0
        @test isnan(besselj1(NaN))
    end

    @testset "Float16" begin
        for x in Float16[0.0, 0.5, 1.3, 2.5, -1.3]
            got = besselj1(x)
            @test got isa Float16
            @test got ≈ Bessels.besselj1(x) rtol = 1.0f-3
        end
    end

    @testset "Float64 lookup tables are Vec-gather-safe" begin
        # J1_ROOTS_HI_F64/J1_ROOTS_LO_F64 and each element of J1_POLYS_F64
        # must be plain `Array` (not Tuple, not StaticArrays.SArray) of
        # length 16 -- one entry per pi/2-wide interval. VectorizationBase
        # defines a real getindex(A::Array, i::AbstractSIMD, ...) method
        # that lowers to a genuine hardware gather -- declared for `Array`
        # specifically because gathering needs a real strided pointer,
        # which neither `Tuple` nor `SArray` have. Indexing a `Tuple` with
        # a `VectorizationBase.Vec` (reached when an AiryPSF{Float64} hits
        # a @turbo evaluate.() call) instead falls into Base's deprecated,
        # non-AbstractVector getindex fallback and recurses indefinitely
        # (StackOverflowError, "program state may be corrupted"). The
        # length check guards against a silent incomplete transpose (all 16
        # intervals must survive the row -> structure-of-arrays reshape).
        # See src/bessels.jl for the full story.
        @test CrowdPhot.J1_ROOTS_HI_F64 isa Array
        @test CrowdPhot.J1_ROOTS_LO_F64 isa Array
        @test length(CrowdPhot.J1_ROOTS_HI_F64) == 16
        @test length(CrowdPhot.J1_ROOTS_LO_F64) == 16
        @test length(CrowdPhot.J1_POLYS_F64) == 14
        @test all(col -> col isa Array && length(col) == 16, CrowdPhot.J1_POLYS_F64)
    end
end
