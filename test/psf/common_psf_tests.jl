using CrowdPhot.PSF: CircularGaussianPSF, GaussianPSF, CircularGaussianPRF, GaussianPRF, CircularMoffatPSF, MoffatPSF, evaluate, centroid, integral, evaluate_fg, evaluate_fgh, AbstractPSFModel, extent, render, theta, amplitude, background, fwhm, peak, effective_area, AiryPSF, ImagePSF, add_star!, subtract_star!
using Test

# Tests generic API, type return, etc
function test_common(model::AbstractPSFModel{T}) where {T}
    # Evaluation
    @test @inferred(evaluate(model, centroid(model)...)) isa T
    cy, cx = round.(Int, centroid(model))
    @test @inferred(evaluate(model, CartesianIndex(cy, cx))) isa T
    @test model(centroid(model)...) ≈ evaluate(model, centroid(model)...)
    @test model(CartesianIndex(cy, cx)) ≈ evaluate(model, CartesianIndex(cy, cx))
    ex = @inferred extent(model)
    @test ex isa Tuple{Tuple{T, T}, Tuple{T, T}}
    y, x = range(ex[1][1], ex[1][2]; step = one(T)), range(ex[2][1], ex[2][2]; step = one(T))
    ex_round = @inferred extent(Int, model)
    @test ex_round == ((floor(Int, ex[1][1]), ceil(Int, ex[1][2])), (floor(Int, ex[2][1]), ceil(Int, ex[2][2])))
    m = evaluate.(model, y, x')
    @test m isa Matrix{T}
    @test size(m) == (length(y), length(x))
    y, x = ex_round[1][1]:ex_round[1][2], ex_round[2][1]:ex_round[2][2]
    m = evaluate.(model, CartesianIndices((y, x)))
    @test m isa Matrix{T}
    @test size(m) == (length(y), length(x))
    @test @inferred(render(model)) isa Matrix{T}
    # Check rendering into a larger image
    image = zeros(T, 20, 20)
    inds = CartesianIndices(model)
    add_star!(image, model, inds)
    for i in inds
        checkbounds(Bool, image, i) || continue
        @test image[i] == evaluate(model, i)
    end
    # Check that doing it again accumulates flux
    add_star!(image, model, inds)
    for i in inds
        checkbounds(Bool, image, i) || continue
        @test image[i] == 2 * evaluate(model, i)
    end
    # Check automatic inds
    fill!(image, 0)
    add_star!(image, model)
    for i in inds
        checkbounds(Bool, image, i) || continue
        @test image[i] == evaluate(model, i)
    end
    subtract_star!(image, model)
    @test all(iszero, image)


    # API functions
    @test @inferred(centroid(model)) isa Tuple{T, T}
    @test @inferred(integral(model)) isa T
    @test @inferred(peak(model)) isa T
    @test @inferred(amplitude(model)) isa T
    @test @inferred(background(model)) isa T
    # @test effective_area(model) isa T # no generic method yet
    # @test fwhm(model) isa T # no generic method yet
    return @test @inferred(theta(model)) isa T
end

for model in (
        AiryPSF(x = 1.3, y = 2.4, radius = 3.0, flux = 120.0, bkg = 10.0),
        AiryPSF(x = 1.3f0, y = 2.4f0, radius = 3.0f0, flux = 120.0f0, bkg = 10.0f0),
        CircularMoffatPSF(x = 1.3, y = 2.4, α = 3.0, β = 3.5, flux = 120.0, bkg = 10.0),
        CircularMoffatPSF(x = 1.3f0, y = 2.4f0, α = 3.0f0, β = 3.5f0, flux = 120.0f0, bkg = 10.0f0),
        MoffatPSF(x = 2.5, y = 5.0, x_α = 3.0, y_α = 4.0, theta = 35.0, β = 3.5, flux = 120.0, bkg = 10.0),
        MoffatPSF(x = 2.5f0, y = 5.0f0, x_α = 3.0f0, y_α = 4.0f0, theta = 35.0f0, β = 3.5f0, flux = 120.0f0, bkg = 10.0f0),
        CircularGaussianPSF(x = 1.3, y = 2.4, fwhm = 3.0, flux = 120.0, bkg = 10.0),
        CircularGaussianPSF(x = 1.3f0, y = 2.4f0, fwhm = 3.0f0, flux = 120.0f0, bkg = 10.0f0),
        GaussianPSF(x = 2.5, y = 5.0, x_fwhm = 3.0, y_fwhm = 4.0, theta = 35, flux = 120.0, bkg = 10),
        GaussianPSF(x = 2.5f0, y = 5.0f0, x_fwhm = 3.0f0, y_fwhm = 4.0f0, theta = 35.0f0, flux = 120.0f0, bkg = 10.0f0),
        CircularGaussianPRF(x = 1.3, y = 2.4, fwhm = 3.0, flux = 120.0, bkg = 10.0),
        CircularGaussianPRF(x = 1.3f0, y = 2.4f0, fwhm = 3.0f0, flux = 120.0f0, bkg = 10.0f0),
        GaussianPRF(x = 2.5, y = 5.0, x_fwhm = 3.0, y_fwhm = 4.0, theta = 35.0, flux = 120.0, bkg = 10),
        GaussianPRF(x = 2.5f0, y = 5.0f0, x_fwhm = 3.0f0, y_fwhm = 4.0f0, theta = 35.0f0, flux = 120.0f0, bkg = 10.0f0),
        ImagePSF(rand(7, 7); x = 3.0, y = 4.0, flux = 120.0, bkg = 7.0, oversampling = 2, normalize = false),
        ImagePSF(rand(Float32, 7, 7); x = 3.0f0, y = 4.0f0, flux = 120.0f0, bkg = 7.0f0, oversampling = 2, normalize = false),
    )
    @testset "API: $(typeof(model))" begin
        test_common(model)
    end
end

@testset "CircularMoffatPSF" begin
    @testset "constructor promotion" begin
        @test CircularMoffatPSF(x = 1.3, y = 2.4, α = 3.0, β = 2.5, flux = 120.0, bkg = 10) isa CircularMoffatPSF{Float64}
        @test CircularMoffatPSF(x = 1.3f0, y = 2.4f0, α = 3.0f0, β = 2.5f0, flux = 120.0f0, bkg = 10.0f0) isa CircularMoffatPSF{Float32}
        @test CircularMoffatPSF(x = 1, y = 2, α = 3, β = 2, flux = 120, bkg = 10) isa CircularMoffatPSF{Float64}
        @test CircularMoffatPSF(x = BigFloat(1.3), y = BigFloat(2.4), α = BigFloat(3.0), β = BigFloat(2.5), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa CircularMoffatPSF{BigFloat}
    end

    m = CircularMoffatPSF(x = 0, y = 0, α = 5, β = 3, flux = 50, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 50.0
    @test all(x -> isapprox(x, 5.098245285339587), fwhm(m))
    @test effective_area(m) ≈ 98.17477042468103 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 0.0
    r1 = evaluate(m, 1, 2)
    @test r1 isa Float64
    @test r1 ≈ 10.736828440240256 ≈ m(1, 2)
    let (f, g) = evaluate_fg(m, 1, 2)
        @test f ≈ evaluate(m, 1, 2)
        @test g ≈ [0.29473137609610256, 0.14736568804805128, -0.14736568804805128, 0.23407451180546335, 0.014736568804805127, 1.0]
    end
end

@testset "MoffatPSF" begin
    @testset "constructor promotion" begin
        @test MoffatPSF(x = 1.3, y = 2.4, x_α = 3.0, y_α = 4.0, theta = 35, β = 2.5, flux = 120.0, bkg = 10.0) isa MoffatPSF{Float64}
        @test MoffatPSF(x = 1.3f0, y = 2.4f0, x_α = 3.0f0, y_α = 4.0f0, theta = 35.0f0, β = 2.5f0, flux = 120.0f0, bkg = 10.0f0) isa MoffatPSF{Float32}
        @test MoffatPSF(x = 1, y = 2, x_α = 3, y_α = 4, theta = 35, β = 2, flux = 120, bkg = 10) isa MoffatPSF{Float64}
        @test MoffatPSF(x = BigFloat(1.3), y = BigFloat(2.4), x_α = BigFloat(3.0), y_α = BigFloat(4.0), theta = BigFloat(35), β = BigFloat(2.5), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa MoffatPSF{BigFloat}
    end

    m = MoffatPSF(x = 0, y = 0, x_α = 5, y_α = 3, theta = 30, β = 3, flux = 50, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 50.0
    @test all(isapprox.(fwhm(m), (3.058947171203752, 5.098245285339587)))
    @test effective_area(m) ≈ 58.90486225480862 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 30.0
    r1 = evaluate(m, 1, 2)
    @test r1 isa Float64
    @test r1 ≈ 11.224137062459514
    let (f, g) = evaluate_fg(m, 1, 2)
        @test f ≈ evaluate(m, 1, 2)
        @test g ≈ [0.5182597116607032, 0.19412102449902383, -0.0011380924381605808, -0.403981071635931, -0.0022692362992106333, 0.38758058770951537, 0.024482741249190258, 1.0]
    end

    # equal α values and theta=0 reduce to CircularMoffatPSF
    mc = CircularMoffatPSF(x = 1.5, y = 2.5, α = 8, β = 2.5, flux = 3, bkg = 0)
    mm = MoffatPSF(x = 1.5, y = 2.5, x_α = 8, y_α = 8, theta = 0, β = 2.5, flux = 3, bkg = 0)
    @test evaluate(mc, 3, 4) ≈ evaluate(mm, 3, 4)
end

# Test specific models; verify return values
@testset "CircularGaussianPSF" begin
    @testset "constructor promotion" begin
        @test CircularGaussianPSF(x = 1.3, y = 2.4, fwhm = 3.0, flux = 120.0, bkg = 10) isa CircularGaussianPSF{Float64}
        @test CircularGaussianPSF(x = 1.3f0, y = 2.4f0, fwhm = 3.0f0, flux = 120.0f0, bkg = 10.0f0) isa CircularGaussianPSF{Float32}
        @test CircularGaussianPSF(x = 1, y = 2, fwhm = 3, flux = 120, bkg = 10) isa CircularGaussianPSF{Float64}
        @test CircularGaussianPSF(x = BigFloat(1.3), y = BigFloat(2.4), fwhm = BigFloat(3.0), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa CircularGaussianPSF{BigFloat}
    end

    m = CircularGaussianPSF(x = 0, y = 0, fwhm = 10, flux = 1, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 1.0
    @test fwhm(m) == (10.0, 10.0)
    @test effective_area(m) ≈ 226.6180070913597 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 0.0
    r1 = evaluate(m, 1, 2)
    @test r1 isa Float64
    @test r1 ≈ 10.0076829778398427705 ≈ m(1, 2)
    let (f, g) = evaluate_fg(m, 1, 2)
        @test f ≈ evaluate(m, 1, 2)
        @test g ≈ [0.000852069508478649, 0.0004260347542393245, -0.0013235781908488918, 0.0076829778398427705, 1.0]
    end
    let (f, g, h) = evaluate_fgh(m, 1, 2)
        @test f ≈ evaluate(m, 1, 2)
        @test g ≈ [0.000852069508478649, 0.0004260347542393245, -0.0013235781908488918, 0.0076829778398427705, 1.0]
        @test h ≈ [-0.00033153722184843266 4.724876619544591e-5 -0.00031720342029373656 0.0008520695084786488 0.0; 4.724876619544591e-5 -0.0004024103711416015 -0.00015860171014686828 0.0004260347542393244 0.0; -0.00031720342029373656 -0.00015860171014686828 0.00031777260218123345 -0.0013235781908488918 0.0; 0.0008520695084786488 0.0004260347542393244 -0.0013235781908488918 0.0 0.0; 0.0 0.0 0.0 0.0 0.0]
    end
end

@testset "GaussianPSF" begin
    @testset "constructor promotion" begin
        @test GaussianPSF(x = 1.3, y = 2.4, x_fwhm = 3.0, y_fwhm = 4.0, theta = 35, flux = 120.0, bkg = 10.0) isa GaussianPSF{Float64}
        @test GaussianPSF(x = 1.3f0, y = 2.4f0, x_fwhm = 3.0f0, y_fwhm = 4.0f0, theta = 35.0f0, flux = 120.0f0, bkg = 10.0f0) isa GaussianPSF{Float32}
        @test GaussianPSF(x = 1, y = 2, x_fwhm = 3, y_fwhm = 4, theta = 35, flux = 120, bkg = 10) isa GaussianPSF{Float64}
        @test GaussianPSF(x = BigFloat(1.3), y = BigFloat(2.4), x_fwhm = BigFloat(3.0), y_fwhm = BigFloat(4.0), theta = BigFloat(35), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa GaussianPSF{BigFloat}
    end

    m = GaussianPSF(x = 0, y = 0, x_fwhm = 10, y_fwhm = 6, theta = 30, flux = 1, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 1.0
    @test fwhm(m) == (6.0, 10.0)
    @test effective_area(m) ≈ 135.9708042548158 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 30.0
    r1 = evaluate(m, 1, 2)
    @test r1 isa Float64
    @test r1 ≈ 10.012793639217442
    let (f, g) = evaluate_fg(m, 1, 2)
        @test f ≈ evaluate(m, 1, 2)
        @test g ≈ [0.0015033449677925976, 0.0005630977264048036, -0.0009259222931891721, -0.0021263779735002405, -6.58250080875286e-6, 0.012793639217441581, 1.0]
    end
    let (f, g, h) = evaluate_fgh(m, 1, 2)
        @test f ≈ evaluate(m, 1, 2)
        @test g ≈ [0.0015033449677925976, 0.0005630977264048036, -0.0009259222931891721, -0.0021263779735002405, -6.58250080875286e-6, 0.012793639217441581, 1.0]
        @test h ≈ [-0.0008480783301008471 0.0006122875136769376 -0.00038306999261779616 -0.00029386735395409803 2.634671600384182e-5 0.0015033449677925974 0.0; 0.0006122875136769376 -0.0016305525409677744 -0.0001991018141163724 -1.73754252774309e-5 -4.13651345345382e-5 0.0005630977264048036 0.0; -0.00038306999261779616 -0.0001991018141163724 8.89162742440947e-5 0.0001538937229624341 -2.641317692779329e-7 -0.0009259222931891721 0.0; -0.00029386735395409803 -1.73754252774309e-5 0.0001538937229624341 0.0007058477596151256 4.522436130571749e-6 -0.0021263779735002405 0.0; 2.634671600384182e-5 -4.13651345345382e-5 -2.641317692779329e-7 4.522436130571749e-6 -1.90375252767113e-6 -6.5825008087528594e-6 0.0; 0.0015033449677925974 0.0005630977264048036 -0.0009259222931891721 -0.0021263779735002405 -6.5825008087528594e-6 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    end
    # equal fwhm + theta=0 reduces to CircularGaussianPSF
    mc = CircularGaussianPSF(x = 1.5, y = 2.5, fwhm = 8, flux = 3, bkg = 0)
    mg = GaussianPSF(x = 1.5, y = 2.5, x_fwhm = 8, y_fwhm = 8, theta = 0, flux = 3, bkg = 0)
    @test evaluate(mc, 3, 4) ≈ evaluate(mg, 3, 4)
end

@testset "CircularGaussianPRF" begin
    @testset "constructor promotion" begin
        @test CircularGaussianPRF(x = 1.3, y = 2.4, fwhm = 3.0, flux = 120.0, bkg = 10) isa CircularGaussianPRF{Float64}
        @test CircularGaussianPRF(x = 1.3f0, y = 2.4f0, fwhm = 3.0f0, flux = 120.0f0, bkg = 10.0f0) isa CircularGaussianPRF{Float32}
        @test CircularGaussianPRF(x = 1, y = 2, fwhm = 3, flux = 120, bkg = 10) isa CircularGaussianPRF{Float64}
        @test CircularGaussianPRF(x = BigFloat(1.3), y = BigFloat(2.4), fwhm = BigFloat(3.0), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa CircularGaussianPRF{BigFloat}
    end

    m = CircularGaussianPRF(x = 0, y = 0, fwhm = 10, flux = 1, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 1.0
    @test fwhm(m) == (10.0, 10.0)
    @test effective_area(m) ≈ 226.6180070913597 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 0.0
    r1 = evaluate(m, 1, 2)
    @test r1 isa Float64
    @test r1 ≈ 10.007652480658708 ≈ m(1, 2)
    let (f, g) = evaluate_fg(m, 1, 2)
        @test f ≈ evaluate(m, 1, 2)
        @test g ≈ [0.0008447735386125179, 0.0004223864695284576, -0.0013132200987741396, 0.007652480658708134, 1.0]
    end
end

@testset "GaussianPRF" begin
    @testset "constructor promotion" begin
        @test GaussianPRF(x = 1.3, y = 2.4, x_fwhm = 3.0, y_fwhm = 4.0, theta = 35.0, flux = 120.0, bkg = 10.0) isa GaussianPRF{Float64}
        @test GaussianPRF(x = 1.3f0, y = 2.4f0, x_fwhm = 3.0f0, y_fwhm = 4.0f0, theta = 35.0f0, flux = 120.0f0, bkg = 10.0f0) isa GaussianPRF{Float32}
        @test GaussianPRF(x = 1, y = 2, x_fwhm = 3, y_fwhm = 4, theta = 35, flux = 120, bkg = 10) isa GaussianPRF{Float64}
        @test GaussianPRF(x = BigFloat(1.3), y = BigFloat(2.4), x_fwhm = BigFloat(3.0), y_fwhm = BigFloat(4.0), theta = BigFloat(35.0), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa GaussianPRF{BigFloat}
    end

    m = GaussianPRF(x = 0, y = 0, x_fwhm = 10, y_fwhm = 6, theta = 55.0, flux = 1, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 1.0
    @test fwhm(m) == (6.0, 10.0)
    @test effective_area(m) ≈ 135.9708042548158 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 55.0
    r1 = evaluate(m, 1, 2)
    @test r1 isa Float64
    @test r1 ≈ 10.012023392514827
    let (f, g) = evaluate_fg(m, 1, 2)
        @test f ≈ evaluate(m, 1, 2)
        @test g ≈ [0.002343118899476432, -4.764880291087338e-5, -0.0009413918015380413, -0.0016373009738971116, -4.255839655242698e-5, 0.012023392514827206, 1.0]
    end

    # equal x/y fwhm with theta=0 collapses to CircularGaussianPRF
    mc = CircularGaussianPRF(x = 1.5, y = 2.5, fwhm = 8, flux = 3, bkg = 0)
    mg = GaussianPRF(x = 1.5, y = 2.5, x_fwhm = 8, y_fwhm = 8, theta = 0, flux = 3, bkg = 0)
    @test evaluate(mc, 3, 4) ≈ evaluate(mg, 3, 4)
end

@testset "AiryPSF" begin
    @testset "constructor promotion" begin
        @test AiryPSF(x = 1.3, y = 2.4, radius = 3.0, flux = 120.0, bkg = 10) isa AiryPSF{Float64}
        @test AiryPSF(x = 1.3f0, y = 2.4f0, radius = 3.0f0, flux = 120.0f0, bkg = 10.0f0) isa AiryPSF{Float32}
        @test AiryPSF(x = 1, y = 2, radius = 3, flux = 120, bkg = 10) isa AiryPSF{Float64}
        @test AiryPSF(x = BigFloat(1.3), y = BigFloat(2.4), radius = BigFloat(3.0), flux = BigFloat(120.0), bkg = BigFloat(10.0)) isa AiryPSF{BigFloat}
    end

    m = AiryPSF(x = 0, y = 0, radius = 10, flux = 50, bkg = 10)
    @test centroid(m) == (0.0, 0.0)
    @test integral(m) == 50.0
    @test evaluate(m, centroid(m)...) ≈ peak(m) # Ensure r=0 is correct, as this is a special case in the code
    @test all(x -> isapprox(x, 8.436659602162363; rtol = 1.0e-6), fwhm(m))
    @test effective_area(m) ≈ 186.21997265876772 rtol = 1.0e-6
    @test background(m) == 10.0
    @test peak(m) ≈ amplitude(m) + background(m)
    @test theta(m) == 0.0
    r1 = evaluate(m, 1, 2)
    @test r1 isa Float64
    @test r1 ≈ 10.484822848946342 ≈ m(1, 2)
    let (f, g) = evaluate_fg(m, 1, 2)
        @test f ≈ evaluate(m, 1, 2)
        @test g ≈ [0.07346384950430244, 0.03673192475215122, -0.0785986074131928, 0.00969645697892684, 1.0]
    end
end

# ---------------------------------------------------------------------------
# render — odd-size guarantee and centering
# ---------------------------------------------------------------------------
@testset "render" begin

    @testset "odd-size guarantee across sub-pixel centroids and FWHMs" begin
        for x0 in (10.0, 10.1, 10.3, 10.5, 10.7, 10.9)
            for fwhm in (2.0, 3.0, 5.0, 7.2)
                m = CircularGaussianPSF(; x=x0, y=20.4, fwhm, flux=100.0, bkg=0.0)
                kern = render(m)
                sz = size(kern)
                @test isodd(sz[1]) && isodd(sz[2])
                cr, cc = sz[1] ÷ 2 + 1, sz[2] ÷ 2 + 1
                maxval, maxidx = findmax(kern)
                @test maxidx == CartesianIndex(cr, cc)
            end
        end
    end

    @testset "symmetry about center for integer-centroid model" begin
        # A circular Gaussian centered exactly on a pixel should produce a
        # rendered kernel that is symmetric about its center pixel.
        for fwhm in (3.0, 5.0, 8.0)
            m = CircularGaussianPSF(; x=15.0, y=25.0, fwhm, flux=50.0, bkg=0.0)
            kern = render(m)
            cr, cc = size(kern, 1) ÷ 2 + 1, size(kern, 2) ÷ 2 + 1
            for r in 1:size(kern, 1), c in 1:size(kern, 2)
                dr, dc = r - cr, c - cc
                @test kern[cr + dr, cc + dc] ≈ kern[cr - dr, cc - dc]
                @test kern[cr + dr, cc + dc] ≈ kern[cr - dr, cc + dc]
                @test kern[cr + dr, cc + dc] ≈ kern[cr + dr, cc - dc]
            end
        end
    end

    @testset "integral conservation (well-sampled PSF)" begin
        # For a well-sampled Gaussian (FWHM ≫ 1 px), the sum of the rendered
        # kernel should approximate the true flux (pixel area = 1.0).
        for fwhm in (4.0, 6.0, 10.0)
            # Place centroid at half-pixel offset — worst case for alignment.
            m = CircularGaussianPSF(; x=15.5, y=25.5, fwhm, flux=100.0, bkg=0.0)
            kern = render(m)
            @test sum(kern) ≈ 100.0 rtol = 0.01
        end
    end

    @testset "extent is fully covered" begin
        # The rendered kernel must cover the full floating-point extent
        # returned by `extent(model)`.
        for (x0, y0) in ((10.0, 20.0), (10.3, 20.7), (10.6, 20.4))
            for fwhm in (2.5, 5.0)
                m = CircularGaussianPSF(; x=x0, y=y0, fwhm, flux=100.0, bkg=0.0)
                (y_lo, y_hi), (x_lo, x_hi) = extent(m)
                kern = render(m)
                cr, cc = size(kern, 1) ÷ 2 + 1, size(kern, 2) ÷ 2 + 1
                hy = cr - 1
                hx = cc - 1
                xc = round(Int, x0)
                yc = round(Int, y0)
                @test xc - hx ≤ x_lo
                @test xc + hx ≥ x_hi
                @test yc - hy ≤ y_lo
                @test yc + hy ≥ y_hi
            end
        end
    end
end
