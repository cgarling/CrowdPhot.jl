using CrowdPhot: calc_total_error
using StableRNGs
using Test

@testset "calc_total_error" begin
    rng = StableRNG(42)

    # -----------------------------------------------------------------------
    # Basic arithmetic
    # -----------------------------------------------------------------------

    @testset "basic" begin
        # data > 0, gain > 0: both terms contribute
        @test calc_total_error(2.0, 1.0, 2.0) ≈ sqrt(1.0^2 + 2.0 / 2.0)
        # data > 0, gain > 0: general case
        @test calc_total_error(5.0, 3.0, 1.0) ≈ sqrt(3.0^2 + 5.0 / 1.0)
        # data <= 0: source Poisson term omitted
        @test calc_total_error(-1.0, 0.5, 2.0) ≈ 0.5
        @test calc_total_error(0.0, 1.5, 2.0) ≈ 1.5
        # gain <= 0: source Poisson term omitted
        @test calc_total_error(10.0, 2.0, 0.0) ≈ 2.0
    end

    # -----------------------------------------------------------------------
    # Effective gain validation
    # -----------------------------------------------------------------------

    @testset "gain validation" begin
        @test_throws ArgumentError calc_total_error(1.0, 1.0, -1.0)
        @test_throws ArgumentError calc_total_error(1.0, 1.0, -1e-6)
        # Zero and positive gains are valid
        @test calc_total_error(1.0, 1.0, 0.0) ≈ 1.0
        @test calc_total_error(1.0, 1.0, 1e-6) ≈ sqrt(1.0 + 1.0 / 1e-6)
    end

    # -----------------------------------------------------------------------
    # Zero-gain pixels (array case)
    # -----------------------------------------------------------------------

    @testset "zero gain" begin
        bkg = ones(5)
        data = fill(2.0, 5)
        gain = fill(2.0, 5)
        gain[1] = 0.0
        gain[3] = 0.0
        result = calc_total_error.(data, bkg, gain)
        # pixels with gain == 0 get bkg_error only
        @test result[1] ≈ 1.0
        @test result[3] ≈ 1.0
        # pixels with gain > 0 include source term
        expected = sqrt(1.0^2 + 2.0 / 2.0)
        @test result[2] ≈ expected
        @test result[4] ≈ expected
        @test result[5] ≈ expected
    end

    # -----------------------------------------------------------------------
    # Negative-data pixels (array case)
    # -----------------------------------------------------------------------

    @testset "negative data" begin
        bkg = [0.5, 1.0, 2.0, 1.5]
        data = [-1.0, 0.0, 1.0, 10.0]
        gain = 2.0
        result = calc_total_error.(data, bkg, gain)
        # data <= 0 → bkg-only
        @test result[1] ≈ 0.5
        @test result[2] ≈ 1.0
        # data > 0 → both terms
        @test result[3] ≈ sqrt(2.0^2 + 1.0 / 2.0)
        @test result[4] ≈ sqrt(1.5^2 + 10.0 / 2.0)
    end

    # -----------------------------------------------------------------------
    # Scalar gain with array data
    # -----------------------------------------------------------------------

    @testset "scalar gain" begin
        data = [1.0, 2.0, 4.0]
        bkg = [0.5, 0.5, 0.5]
        result = calc_total_error.(data, bkg, 2.0)
        for (d, r) in zip(data, result)
            @test r ≈ sqrt(0.5^2 + d / 2.0)
        end
    end

    # -----------------------------------------------------------------------
    # Broadcasting over matrices
    # -----------------------------------------------------------------------

    @testset "matrix broadcast" begin
        data = fill(2.0, 4, 4)
        bkg = ones(4, 4)
        gain = fill(2.0, 4, 4)
        result = calc_total_error.(data, bkg, gain)
        @test size(result) == (4, 4)
        @test all(r -> r ≈ sqrt(2.0), result)
    end

    # -----------------------------------------------------------------------
    # Type stability and promotion
    # -----------------------------------------------------------------------

    @testset "types" begin
        # Float64
        r64 = calc_total_error(2.0, 1.0, 2.0)
        @test r64 isa Float64
        # Float32
        r32 = calc_total_error(2.0f0, 1.0f0, 2.0f0)
        @test r32 isa Float32
        @test r32 ≈ sqrt(Float32(1.0^2 + 2.0 / 2.0))
        # Mixed Float32/Float64 promotes
        r_mix = calc_total_error(2.0f0, 1.0, 2.0f0)
        @test r_mix ≈ sqrt(1.0 + 2.0 / 2.0)
    end

    # -----------------------------------------------------------------------
    # Agreement with photutils reference values
    # -----------------------------------------------------------------------

    @testset "photutils agreement" begin
        # Reference values computed with photutils 1.x
        data      = [2.0, -1.0, 10.0, 0.0, 5.0]
        bkg_error = [1.0, 0.5, 2.0, 1.5, 3.0]
        effgain   = [2.0, 2.0, 0.0, 4.0, 1.0]
        ref = [sqrt(1^2 + 2/2), 0.5, 2.0, 1.5, sqrt(3^2 + 5/1)]
        result = calc_total_error.(data, bkg_error, effgain)
        @test result ≈ ref atol = 1e-15
    end

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------

    @testset "edge cases" begin
        # Very large gain → Poisson term negligible
        @test calc_total_error(1.0, 1.0, 1e12) ≈ 1.0 atol = 1e-12
        # Very small gain → Poisson term dominates
        r = calc_total_error(1.0, 0.0, 1e-12)
        @test r ≈ sqrt(1.0 / 1e-12)
        # Negative bkg_error (should still give positive error)
        @test calc_total_error(1.0, -2.0, 0.0) ≈ 2.0
        @test calc_total_error(1.0, -2.0, 2.0) ≈ sqrt(4.0 + 0.5)
    end
end
