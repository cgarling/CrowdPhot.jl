using CrowdPhot:
    CircularAperture,
    bounding_axes,
    clipped_axes,
    _overlap_flag,
    aperture_weight,
    ExactOverlap,
    CenterOverlap,
    WholePixelOverlap,
    SubpixelOverlap,
    inside, outside, partial, OverlapFlag
using OffsetArrays
using Test

# ==============================================================================
# Construction
# ==============================================================================

@testset "CircularAperture construction" begin
    ap = CircularAperture(y=10.0, x=15.0, r=3.0)
    @test ap.y == 10.0
    @test ap.x == 15.0
    @test ap.r == 3.0
    @test ap isa CircularAperture{Float64}

    # Positional constructor
    ap2 = CircularAperture(10.0, 15.0, 3.0)
    @test ap2.y == ap.y && ap2.x == ap.x && ap2.r == ap.r

    # Integer arguments produce integer type parameter (consistent with @kwdef convention)
    ap3 = CircularAperture(y=10, x=15, r=3)
    @test ap3 isa CircularAperture{Int}
    @test ap3.y == 10 && ap3.x == 15 && ap3.r == 3

    # Mixed types promote
    ap4 = CircularAperture(y=10, x=15.0, r=3)
    @test ap4 isa CircularAperture{Float64}
    @test ap4.y == 10.0 && ap4.x == 15.0 && ap4.r == 3.0
end

# ==============================================================================
# bounding_axes
# ==============================================================================

@testset "bounding_axes" begin
    @testset "on-grid center, integer radius" begin
        ap = CircularAperture(y=10.0, x=15.0, r=3.0)
        yr, xr = bounding_axes(ap)
        # center ± (r + 0.5) → 10 ± 3.5 → [6.5, 13.5] → 7:13
        @test yr == 7:13
        @test xr == 12:18
        @test length(yr) == 7
        @test length(xr) == 7
    end

    @testset "off-grid center" begin
        ap = CircularAperture(y=10.3, x=15.7, r=2.0)
        yr, xr = bounding_axes(ap)
        # The bounding axes must include all pixels with possible overlap.
        # A pixel at (i, j) can overlap iff |i - yc| <= r + 0.5 and
        # |j - xc| <= r + 0.5 (Chebyshev bound), so the range must extend
        # at least to yc - r - 0.5 and yc + r + 0.5.
        @test first(yr) <= 10.3 - 2.0 + 0.5
        @test last(yr) >= 10.3 + 2.0 - 0.5
        @test first(xr) <= 15.7 - 2.0 + 0.5
        @test last(xr) >= 15.7 + 2.0 - 0.5
    end

    @testset "zero radius" begin
        ap = CircularAperture(y=5.0, x=5.0, r=0.0)
        yr, xr = bounding_axes(ap)
        # pixel at (5,5) with r=0: _ymin = 5 - 0 - 0.5 = 4.5, ceil(4.5) = 5
        # ymax = ceil(5 + 0 - 0.5) = ceil(4.5) = 5
        @test yr == 5:5
        @test xr == 5:5
    end
end

# ==============================================================================
# clipped_axes
# ==============================================================================

@testset "clipped_axes" begin
    @testset "fully inside image" begin
        ap = CircularAperture(y=10.0, x=10.0, r=3.0)
        img = zeros(20, 20)
        yr, xr = clipped_axes(ap, img)
        @test yr == bounding_axes(ap)[1]
        @test xr == bounding_axes(ap)[2]
    end

    @testset "partially outside image" begin
        ap = CircularAperture(y=2.0, x=10.0, r=5.0)
        img = zeros(20, 20)
        yr, xr = clipped_axes(ap, img)
        @test first(yr) == 1        # clamped to image top
        @test first(xr) >= 1
        @test last(xr) <= 20
    end

    @testset "fully outside image" begin
        ap = CircularAperture(y=100.0, x=100.0, r=3.0)
        img = zeros(20, 20)
        yr, xr = clipped_axes(ap, img)
        # An empty range: first > last
        @test isempty(yr) || isempty(xr)
    end

    @testset "offset image axes (non-1-based)" begin
        ap = CircularAperture(y=5.0, x=5.0, r=3.0)
        img = OffsetArrays.Origin(10, 10)(zeros(20, 20))
        yr, xr = clipped_axes(ap, img)
        @test first(yr) >= 10
        @test first(xr) >= 10
    end
end

# ==============================================================================
# _overlap_flag
# ==============================================================================

@testset "_overlap_flag" begin
    ap = CircularAperture(y=10.0, x=15.0, r=3.0)

    @testset "inside classification" begin
        # Center pixel is always inside for r > 0
        @test _overlap_flag(ap, 10, 15) == inside
        # Pixel well within circle
        @test _overlap_flag(ap, 11, 16) == inside
    end

    @testset "outside classification" begin
        @test _overlap_flag(ap, 1, 1) == outside
        @test _overlap_flag(ap, 20, 30) == outside
    end

    @testset "partial classification" begin
        # Pixels near the boundary should be partial
        flag = _overlap_flag(ap, 7, 15)  # near top edge of circle
        @test flag == partial || flag == outside
        flag2 = _overlap_flag(ap, 10, 12)  # near left edge of circle
        @test flag2 == partial || flag2 == outside
    end

    @testset "consistency with aperture_weight" begin
        ap2 = CircularAperture(y=20.0, x=20.0, r=5.0)
        yr, xr = bounding_axes(ap2)
        for j in xr, i in yr
            flag = _overlap_flag(ap2, i, j)
            w = aperture_weight(ap2, i, j, ExactOverlap())
            if flag == outside
                @test w == 0.0
            elseif flag == inside
                @test w == 1.0
            end
        end
    end
end

# ==============================================================================
# aperture_weight — ExactOverlap
# ==============================================================================

@testset "aperture_weight ExactOverlap" begin
    @testset "sum of weights equals πr²" begin
        for r in (1.0, 2.0, 3.5, 5.0, 7.3)
            ap = CircularAperture(y=15.0, x=15.0, r=r)
            yr, xr = bounding_axes(ap)
            total = sum(aperture_weight(ap, i, j, ExactOverlap()) for j in xr, i in yr)
            @test total ≈ π * r^2 rtol=1e-12
        end
    end

    @testset "constant-image aperture sum equals area × constant" begin
        ap = CircularAperture(y=10.0, x=10.0, r=3.0)
        img = fill(2.0, 20, 20)
        yr, xr = clipped_axes(ap, img)
        weighted_sum = zero(Float64)
        for j in xr, i in yr
            w = aperture_weight(ap, i, j, ExactOverlap())
            weighted_sum += w * img[i, j]
        end
        @test weighted_sum ≈ π * 3.0^2 * 2.0 rtol=1e-12
    end

    @testset "zero radius returns zero" begin
        ap = CircularAperture(y=10.0, x=10.0, r=0.0)
        @test aperture_weight(ap, 10, 10, ExactOverlap()) == 0.0
        @test aperture_weight(ap, 9, 10, ExactOverlap()) == 0.0
    end

    @testset "weights are in [0, 1]" begin
        ap = CircularAperture(y=15.0, x=15.0, r=4.0)
        yr, xr = bounding_axes(ap)
        for j in xr, i in yr
            w = aperture_weight(ap, i, j, ExactOverlap())
            @test 0.0 <= w <= 1.0
        end
    end
end

# ==============================================================================
# aperture_weight — CenterOverlap
# ==============================================================================

@testset "aperture_weight CenterOverlap" begin
    @testset "center pixel is always 1.0 for r > 0" begin
        ap = CircularAperture(y=10.0, x=10.0, r=3.0)
        @test aperture_weight(ap, 10, 10, CenterOverlap()) == 1.0
    end

    @testset "binary output" begin
        ap = CircularAperture(y=15.0, x=15.0, r=5.0)
        yr, xr = bounding_axes(ap)
        for j in xr, i in yr
            w = aperture_weight(ap, i, j, CenterOverlap())
            @test (w == 0.0 || w == 1.0)
        end
    end

    @testset "area >= WholePixelOverlap area" begin
        ap = CircularAperture(y=20.0, x=20.0, r=5.0)
        yr, xr = bounding_axes(ap)
        cnt_center = sum(aperture_weight(ap, i, j, CenterOverlap()) for j in xr, i in yr)
        cnt_whole  = sum(aperture_weight(ap, i, j, WholePixelOverlap()) for j in xr, i in yr)
        @test cnt_center >= cnt_whole
    end
end

# ==============================================================================
# aperture_weight — WholePixelOverlap
# ==============================================================================

@testset "aperture_weight WholePixelOverlap" begin
    @testset "binary output" begin
        ap = CircularAperture(y=15.0, x=15.0, r=5.0)
        yr, xr = bounding_axes(ap)
        for j in xr, i in yr
            w = aperture_weight(ap, i, j, WholePixelOverlap())
            @test (w == 0.0 || w == 1.0)
        end
    end

    @testset "pixel crossing boundary is excluded" begin
        # r=0.3: pixel half-diagonal ≈ 0.707 > 0.3, so no pixel is wholly inside
        ap = CircularAperture(y=10.0, x=10.0, r=0.3)
        @test aperture_weight(ap, 10, 10, WholePixelOverlap()) == 0.0
    end

    @testset "wholly-inside pixel is included" begin
        # radius large enough that center pixel corners are all inside
        ap = CircularAperture(y=10.0, x=10.0, r=2.0)
        # Center pixel corners are at (±0.5, ±0.5) from center; max distance = sqrt(0.5) ≈ 0.707 < 2.0
        @test aperture_weight(ap, 10, 10, WholePixelOverlap()) == 1.0
    end
end

# ==============================================================================
# aperture_weight — SubpixelOverlap
# ==============================================================================

@testset "aperture_weight SubpixelOverlap" begin
    @testset "converges to ExactOverlap as N increases" begin
        ap = CircularAperture(y=20.0, x=20.0, r=3.5)
        yr, xr = bounding_axes(ap)
        exact_vals = [aperture_weight(ap, i, j, ExactOverlap()) for j in xr, i in yr]
        sub5_vals  = [aperture_weight(ap, i, j, SubpixelOverlap{5}()) for j in xr, i in yr]
        sub20_vals = [aperture_weight(ap, i, j, SubpixelOverlap{20}()) for j in xr, i in yr]
        max_err_5  = maximum(abs.(exact_vals - sub5_vals))
        max_err_20 = maximum(abs.(exact_vals - sub20_vals))
        @test max_err_20 < max_err_5 * 0.5  # N=20 is noticeably better than N=5
    end

    @testset "SubpixelOverlap{1} equals CenterOverlap" begin
        ap = CircularAperture(y=10.0, x=10.0, r=3.0)
        yr, xr = bounding_axes(ap)
        for j in xr, i in yr
            w_sub1 = aperture_weight(ap, i, j, SubpixelOverlap{1}())
            w_cen  = aperture_weight(ap, i, j, CenterOverlap())
            @test w_sub1 == w_cen
        end
    end
end

# ==============================================================================
# Edge cases
# ==============================================================================

@testset "edge cases" begin
    @testset "aperture at image corner" begin
        ap = CircularAperture(y=1.0, x=1.0, r=3.0)
        img = zeros(10, 10)
        yr, xr = clipped_axes(ap, img)
        @test first(yr) == 1
        @test first(xr) == 1
        # Weights should be computable without error
        for j in xr, i in yr
            w = aperture_weight(ap, i, j, ExactOverlap())
            @test isfinite(w)
        end
    end

    @testset "negative radius — ExactOverlap returns 0" begin
        ap = CircularAperture(y=10.0, x=10.0, r=-1.0)
        @test aperture_weight(ap, 10, 10, ExactOverlap()) == 0.0
    end

    @testset "Integer-typed aperture works with all methods" begin
        ap = CircularAperture(y=10, x=15, r=3)
        @test aperture_weight(ap, 10, 15, ExactOverlap()) isa AbstractFloat
        @test aperture_weight(ap, 10, 15, CenterOverlap()) isa AbstractFloat
        @test aperture_weight(ap, 10, 15, WholePixelOverlap()) isa AbstractFloat
        yr, xr = bounding_axes(ap)
        @test yr isa AbstractUnitRange{<:Integer}
        @test xr isa AbstractUnitRange{<:Integer}
    end

    @testset "clipped_axes with empty intersection is non-throwing" begin
        ap = CircularAperture(y=100.0, x=100.0, r=3.0)
        img = zeros(10, 10)
        yr, xr = clipped_axes(ap, img)
        @test isempty(yr) || isempty(xr) || isempty(CartesianIndices((yr, xr)))
    end

    @testset "Integer-typed aperture return type matches float(T)" begin
        # CircularAperture{Int}: float(Int) = Float64, so all weight
        # return types should be Float64.
        ap_i = CircularAperture(y=10, x=15, r=3)
        @test typeof(aperture_weight(ap_i, 10, 15, ExactOverlap())) == Float64
        @test typeof(aperture_weight(ap_i, 10, 15, CenterOverlap())) == Float64
        @test typeof(aperture_weight(ap_i, 10, 15, WholePixelOverlap())) == Float64
        @test typeof(aperture_weight(ap_i, 10, 15, SubpixelOverlap{5}())) == Float64
    end
end

# ==============================================================================
# Float type consistency
# ==============================================================================

@testset "float type consistency" begin
    @testset "Float32 inputs produce Float32 outputs" begin
        ap32 = CircularAperture(y=10f0, x=15f0, r=3f0)
        FT = Float32

        # ExactOverlap: inside, partial, and outside paths
        @test aperture_weight(ap32, 10, 15, ExactOverlap()) isa FT  # inside
        @test aperture_weight(ap32, 1, 1, ExactOverlap()) isa FT    # outside
        # Partial pixel: near the edge of r=3 from center (10, 15)
        @test aperture_weight(ap32, 7, 15, ExactOverlap()) isa FT   # partial

        # CenterOverlap
        @test aperture_weight(ap32, 10, 15, CenterOverlap()) isa FT
        @test aperture_weight(ap32, 1, 1, CenterOverlap()) isa FT

        # WholePixelOverlap
        @test aperture_weight(ap32, 10, 15, WholePixelOverlap()) isa FT
        @test aperture_weight(ap32, 7, 15, WholePixelOverlap()) isa FT

        # SubpixelOverlap
        @test aperture_weight(ap32, 10, 15, SubpixelOverlap{3}()) isa FT
        @test aperture_weight(ap32, 7, 15, SubpixelOverlap{3}()) isa FT

        # Full-sum aggregation preserves type
        yr, xr = bounding_axes(ap32)
        total = sum(aperture_weight(ap32, i, j, ExactOverlap()) for j in xr, i in yr)
        @test total isa FT

        total_center = sum(aperture_weight(ap32, i, j, CenterOverlap()) for j in xr, i in yr)
        @test total_center isa FT
    end

    @testset "Float64 inputs produce Float64 outputs" begin
        ap64 = CircularAperture(y=10.0, x=15.0, r=3.0)
        FT = Float64

        @test aperture_weight(ap64, 10, 15, ExactOverlap()) isa FT
        @test aperture_weight(ap64, 1, 1, ExactOverlap()) isa FT
        @test aperture_weight(ap64, 7, 15, ExactOverlap()) isa FT  # partial
        @test aperture_weight(ap64, 10, 15, CenterOverlap()) isa FT
        @test aperture_weight(ap64, 10, 15, WholePixelOverlap()) isa FT
        @test aperture_weight(ap64, 10, 15, SubpixelOverlap{3}()) isa FT

        yr, xr = bounding_axes(ap64)
        total = sum(aperture_weight(ap64, i, j, ExactOverlap()) for j in xr, i in yr)
        @test total isa FT
    end

    @testset "Float32 and Float64 produce consistent values" begin
        ap32 = CircularAperture(y=10f0, x=15f0, r=3f0)
        ap64 = CircularAperture(y=10.0, x=15.0, r=3.0)

        # Binary methods should agree exactly
        @test aperture_weight(ap32, 10, 15, CenterOverlap()) ==
              aperture_weight(ap64, 10, 15, CenterOverlap())
        @test aperture_weight(ap32, 10, 15, WholePixelOverlap()) ==
              aperture_weight(ap64, 10, 15, WholePixelOverlap())

        # ExactOverlap values should be approximately equal
        for (i, j) in ((10, 15), (11, 16), (8, 15), (7, 13))
            w32 = aperture_weight(ap32, i, j, ExactOverlap())
            w64 = aperture_weight(ap64, i, j, ExactOverlap())
            @test w32 ≈ w64 atol=1e-5
        end

        # Summed area should agree
        yr, xr = bounding_axes(ap64)
        total32 = sum(aperture_weight(ap32, i, j, ExactOverlap()) for j in xr, i in yr)
        total64 = sum(aperture_weight(ap64, i, j, ExactOverlap()) for j in xr, i in yr)
        @test total32 ≈ total64 rtol=1e-5
    end

    @testset "mixed-type construction promotes to float" begin
        # Mixed Int/Float64 → Float64 (no forced float conversion)
        ap_mixed = CircularAperture(y=10, x=15.0, r=3)
        @test ap_mixed isa CircularAperture{Float64}

        # Mixed Float32/Float64 → Float64
        ap_mixed2 = CircularAperture(y=10f0, x=15.0, r=3f0)
        @test ap_mixed2 isa CircularAperture{Float64}

        # All Float32 stays Float32
        ap_f32 = CircularAperture(y=10f0, x=15f0, r=3f0)
        @test ap_f32 isa CircularAperture{Float32}
    end

    @testset "_overlap_flag returns enum regardless of T" begin
        ap32 = CircularAperture(y=10f0, x=15f0, r=3f0)
        ap64 = CircularAperture(y=10.0, x=15.0, r=3.0)
        ap_i = CircularAperture(y=10, x=15, r=3)

        for ap in (ap32, ap64, ap_i)
            @test _overlap_flag(ap, 10, 15) isa OverlapFlag
            @test _overlap_flag(ap, 1, 1) isa OverlapFlag
        end
    end
end
