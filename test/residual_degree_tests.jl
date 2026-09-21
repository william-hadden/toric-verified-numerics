using Test
using IntervalArithmetic

include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))

degree_product(a, b) = (a[1] + b[1], a[2] + b[2])
degree_sum(a, b) = (max(a[1], b[1]), max(a[2], b[2]))
degree_derivative(a, dim) =
    dim == 1 ? (a[1] - 1, a[2]) : (a[1], a[2] - 1)

function residual_factor_degree(q_degree, h_degree, dim)
    return degree_sum(
        degree_derivative(q_degree, dim),
        degree_product(q_degree, h_degree),
    )
end

function rational_csv_shape_and_corner(path)
    rows = 0
    columns = nothing
    last_line = ""

    open(path, "r") do io
        for line in eachline(io)
            rows += 1
            line_columns = count(==(','), line) + 1
            columns === nothing && (columns = line_columns)
            line_columns == columns || error("Coefficient CSV is not rectangular: $path")
            last_line = line
        end
    end

    rows > 0 || error("Coefficient CSV is empty: $path")
    comma = findlast(==(','), last_line)
    corner = comma === nothing ? last_line : last_line[nextind(last_line, comma):end]
    return (rows, columns), corner
end

function rational_string_is_nonzero(entry)
    slash = findfirst(==('/'), entry)
    numerator = strip(slash === nothing ? entry : entry[begin:prevind(entry, slash)])
    return any(character -> isdigit(character) && character != '0', numerator)
end

function float_product(a, b)
    degx = size(a, 1) + size(b, 1) - 2
    degy = size(a, 2) + size(b, 2) - 2
    a_values = cheb_coeffs_to_lobatto_values_2d(cheb_pad(a, degx, degy))
    b_values = cheb_coeffs_to_lobatto_values_2d(cheb_pad(b, degx, degy))
    return cheb_lobatto_values_to_coeffs_2d(a_values .* b_values)
end

function extremal_residual_factors(d)
    G = zeros(Float64, 2d + 3, 2d + 3)
    G[2d + 3, 2d + 1] = 1.0
    G[2d + 2, 2d + 2] = 1.0
    G[2d + 1, 2d + 3] = 1.0

    Hx = zeros(Float64, d, d + 1)
    Hy = zeros(Float64, d + 1, d)
    Hx[end, end] = 1e6
    Hy[end, end] = 1e6
    return G, Hx, Hy
end

function advance_residual_factor(q, h_derivative, dim)
    return cheb_add(
        cheb_trim_exact(partial_coeffs(q, dim)),
        float_product(q, h_derivative),
    )
end

function roundtrip_on_square(coeffs, modes)
    padded = cheb_pad(coeffs, modes - 1, modes - 1)
    values = cheb_coeffs_to_lobatto_values_2d(padded)
    return cheb_lobatto_values_to_coeffs_2d(values)
end

@testset "residual derivative degree allocation" begin
    d = 89
    u = (d, d)
    uxx = degree_derivative(degree_derivative(u, 1), 1)
    uxy = degree_derivative(degree_derivative(u, 1), 2)
    uyy = degree_derivative(degree_derivative(u, 2), 2)

    G = degree_sum(
        degree_product((4, 4), degree_product(uxx, uyy)),
        degree_product((4, 4), degree_product(uxy, uxy)),
    )
    Hx = (d - 1, d)
    Hy = (d, d - 1)

    Qx = residual_factor_degree(G, Hx, 1)
    Qy = residual_factor_degree(G, Hy, 2)
    Qxx = residual_factor_degree(Qx, Hx, 1)
    Qxy = residual_factor_degree(Qx, Hy, 2)
    Qyy = residual_factor_degree(Qy, Hy, 2)

    @test G == (180, 180)
    @test residual_factor_degree(Qxx, Hx, 1) == (444, 447)
    @test residual_factor_degree(Qxx, Hy, 2) == (445, 446)
    @test residual_factor_degree(Qxy, Hy, 2) == (446, 445)
    @test residual_factor_degree(Qyy, Hy, 2) == (447, 444)
end

@testset "degree-89 residual boundary modes" begin
    if isfile(U0_PATH)
        csv_size, corner = rational_csv_shape_and_corner(U0_PATH)
        @test csv_size == (90, 90)
        @test rational_string_is_nonzero(corner)
    else
        @test_skip isfile(U0_PATH)
    end

    @test _CHEB_HAS_FFTW
    if _CHEB_HAS_FFTW
        # These are precisely the extremal modes induced by a nonzero
        # degree-(89,89) input.  Their magnitudes are normalized to avoid
        # losing the fifth-order boundary coefficient in Float64 roundoff.
        G, Hx, Hy = extremal_residual_factors(89)
        Qx = advance_residual_factor(G, Hx, 1)
        Qxx = advance_residual_factor(Qx, Hx, 1)
        Qxxx = advance_residual_factor(Qxx, Hx, 1)

        Qy = advance_residual_factor(G, Hy, 2)
        Qyy = advance_residual_factor(Qy, Hy, 2)
        Qyyy = advance_residual_factor(Qyy, Hy, 2)

        @test size(Qxxx) == (445, 448)
        @test size(Qyyy) == (448, 445)

        Qxxx_square = roundtrip_on_square(Qxxx, 448)
        Qyyy_square = roundtrip_on_square(Qyyy, 448)
        xxx_scale = maximum(abs, Qxxx_square)
        yyy_scale = maximum(abs, Qyyy_square)
        xxx_tail = maximum(abs, @view Qxxx_square[446:448, :])
        yyy_tail = maximum(abs, @view Qyyy_square[:, 446:448])

        @test isfinite(xxx_scale) && xxx_scale > 0
        @test isfinite(yyy_scale) && yyy_scale > 0
        @test xxx_tail <= 1e-10 * xxx_scale
        @test yyy_tail <= 1e-10 * yyy_scale
        @test abs(Qxxx_square[443, 448]) > 0
        @test abs(Qyyy_square[448, 443]) > 0
        @test abs(Qxxx_square[443, 448]) > 1e-4 * xxx_scale
        @test abs(Qyyy_square[448, 443]) > 1e-4 * yyy_scale
        @test abs(Qxxx_square[443, 448]) > 10 * xxx_tail
        @test abs(Qyyy_square[448, 443]) > 10 * yyy_tail
    end
end
