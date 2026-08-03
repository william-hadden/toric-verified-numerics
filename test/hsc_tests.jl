@testset "HSC checkpoint metadata and round trip" begin
    mktempdir() do checkpoint_dir
        coeffs_path = joinpath(checkpoint_dir, "coeffs-rational.csv")
        write(coeffs_path, "1/1,0/1\n")
        context = hsc_checkpoint_context(coeffs_path)
        @test context.coefficient_size == filesize(coeffs_path)
        @test context.bigfloat_precision == precision(BigFloat)
        @test context.coefficient_format == :rational_csv
        parameters = (operation = :inverse_derivatives, k = 2, pdeg = :automatic)
        result = (D = [interval(BigFloat(1))], label = "rational coefficients")

        path = save_hsc_checkpoint(
            4,
            result;
            context,
            parameters,
            checkpoint_dir,
        )

        @test isfile(path)
        @test isequal(
            load_hsc_checkpoint(
                4;
                expected_context = context,
                expected_parameters = parameters,
                checkpoint_dir,
            ),
            result,
        )
        @test_throws ArgumentError load_hsc_checkpoint(
            4;
            expected_context = merge(context, (coefficient_sha256 = "other",)),
            expected_parameters = parameters,
            checkpoint_dir,
        )
        @test_throws ArgumentError load_hsc_checkpoint(
            4;
            expected_context = context,
            expected_parameters = merge(parameters, (k = 3,)),
            checkpoint_dir,
        )

        @test load_hsc_checkpoint_if_valid(
            4;
            expected_context = merge(context, (coefficient_sha256 = "other",)),
            expected_parameters = parameters,
            checkpoint_dir,
        ) === nothing

        @test endswith(hsc_checkpoint_path(5; checkpoint_dir), "step5.jls")
        @test endswith(hsc_checkpoint_path(6; checkpoint_dir), "step6.jls")
        @test_throws ArgumentError hsc_checkpoint_path(7; checkpoint_dir)
    end
end


@testset "HSC run checkpoint folder and README" begin
    mktempdir() do checkpoint_root
        parameters = (
            coefficient_sha256 = "abc123",
            bigfloat_precision = 200,
            pdeg = 10,
            domain = (0, 1, 0, 1),
            eta = 0,
            rho = 0,
            grid = 12,
            direction_order = 5,
            directions_per_cell = 2,
            max_seeds = 100,
            seed_cutoff = 0,
            candidate_shapes = ((0, 0), (1, 1)),
            screen_depth = 0,
            maxdepth = 6,
            beam_width = 20,
            invalid_keep = 5,
        )
        started_at = DateTime(2026, 8, 3, 15, 4, 5)
        first_dir = hsc_run_checkpoint_dir(parameters; checkpoint_root, started_at)
        second_dir = hsc_run_checkpoint_dir(parameters; checkpoint_root, started_at)
        readme = read(joinpath(first_dir, "README.txt"), String)

        @test first_dir != second_dir
        @test dirname(first_dir) == checkpoint_root
        @test occursin(r"_\d{8}-\d{6}$", first_dir)
        @test occursin(r"_\d{8}-\d{6}_2$", second_dir)
        @test occursin("started_at = ", readme)
        @test occursin("pdeg = 10", readme)
        @test occursin("grid = 12", readme)
        @test occursin("maxdepth = 6", readme)

        context = (coefficient_sha256 = "abc123",)
        step_parameters = (operation = :find_compatible_U_V,)
        save_hsc_checkpoint(
            6,
            :older;
            context,
            parameters = step_parameters,
            checkpoint_dir = first_dir,
        )
        save_hsc_checkpoint(
            6,
            :newer;
            context,
            parameters = step_parameters,
            checkpoint_dir = second_dir,
        )
        @test load_hsc_checkpoint(
            6;
            expected_context = context,
            expected_parameters = step_parameters,
            checkpoint_root,
        ) == :newer
    end
end


@testset "HSC pdeg contraction retains discarded tails" begin
    epsilon = big(1) // big(100)
    coeff_matrix(c) = exact_big_interval.([
        c epsilon
        -epsilon epsilon
    ])

    A = [2 1//2; 1//2 3//2]
    full_data = Dict{Tuple{Int, Int, Tuple{Int, Int}}, Any}()
    packed_data = Dict{Tuple{Int, Int, Tuple{Int, Int}}, Any}()
    for i in 1:2, j in 1:2
        key = (i, j, (0, 0))
        full_data[key] = coeff_matrix(A[i, j])
        packed_data[key] = truncated_coeff_enclosure(full_data[key], 1)
        for a in 1:2, b in 1:2
            derivative_key = (i, j, derivative_exponent_multiindex_2d(a, b))
            constant = (i == j ? -2//1 : 1//3) +
                       (a == b ? -1//2 : 1//5)
            full_data[derivative_key] = coeff_matrix(constant)
            packed_data[derivative_key] =
                truncated_coeff_enclosure(full_data[derivative_key], 1)
        end
    end

    derivatives = (
        deriv_num = full_data,
        D = reshape([exact_big_interval(2)], 1, 1),
        pack = (; data = packed_data),
        pdeg = 1,
    )
    ξ = (big(2)//big(1), big(-1)//big(1))
    base = prepare_hsc_base_coeffs(derivatives; pdeg = 1)
    prepared = prepare_hsc_direction_coeffs(derivatives, base, ξ; pdeg = 1)
    compact = compact_curvature_ranges_on_box(
        prepared,
        PolytopeBox(1//4, 3//4, 1//4, 3//4),
    )

    X = exact_big_interval(1//2)
    Y = exact_big_interval(1//2)
    geometry = evaluate_geometry_on_box(derivatives, X, Y)
    direct_Q = compute_Q_rigorously(geometry, ξ, X, Y)
    direct_R2 = riem_sq_range(geometry)

    @test compact !== nothing
    @test inf(compact.Q) <= inf(direct_Q) <= sup(direct_Q) <= sup(compact.Q)
    @test inf(compact.R2) <= inf(direct_R2) <= sup(direct_R2) <= sup(compact.R2)
end


@testset "HSC reuses rigorous bound_metric pdeg pipeline" begin
    coefficient_matrix(c) = exact_big_interval.([
        c 1//100 -1//200
        -1//150 1//250 1//300
        1//400 -1//500 1//600
    ])
    derivatives = (
        uxx = coefficient_matrix(2),
        uxy = coefficient_matrix(1//10),
        uyy = coefficient_matrix(3//2),
    )

    full = build_inverse_metric_coeffs(derivatives)
    value_enclosures = prepare_inverse_coeffs_for_subdivision(full, 2)
    trunc = (
        A11 = enclosure_to_coeffs(value_enclosures.A11),
        A12 = enclosure_to_coeffs(value_enclosures.A12),
        A22 = enclosure_to_coeffs(value_enclosures.A22),
        D = enclosure_to_coeffs(value_enclosures.D),
    )

    @test size(trunc.A11) == (2, 2)
    @test size(trunc.A12) == (2, 2)
    @test size(trunc.A22) == (2, 2)
    @test size(trunc.D) == (2, 2)

    quotient_derivatives =
        compute_inverse_derivative_numerator_components_truncated_coeff_space(
            full;
            k = 2,
            pdeg = 2,
        )
    @test quotient_derivatives.pdeg == 2
    @test all(
        size(enclosure.coeffs) == (2, 2) && enclosure.pdeg == 2
        for enclosure in values(quotient_derivatives.pack.data)
    )
    for enclosure in values(quotient_derivatives.pack.data)
        @test enclosure.tail >= 0
    end

    for (x, y) in ((1//4, 1//3), (1//2, 1//2), (3//4, 2//3))
        X = exact_big_interval(x)
        Y = exact_big_interval(y)
        for field in (:A11, :A12, :A22, :D)
            full_value = evaluate_coeffs_at_point(getproperty(full, field), X, Y)
            trunc_value = evaluate_coeffs_at_point(getproperty(trunc, field), X, Y)
            @test inf(trunc_value) <= inf(full_value)
            @test sup(full_value) <= sup(trunc_value)
        end
    end
end

@testset "HSC coefficient-space curvature contraction" begin
    c(x) = reshape([exact_big_interval(x)], 1, 1)
    enc(x) = truncated_coeff_enclosure(c(x), 1)

    A = [2 1//2; 1//2 3//2]
    N = Dict{Tuple{Int, Int, Int, Int}, Rational{Int}}()
    for i in 1:2, j in 1:2, a in 1:2, b in 1:2
        N[(i, j, a, b)] = (i == j ? -2//1 : 1//3) +
                           (a == b ? -1//2 : 1//5)
    end

    data = Dict{Tuple{Int, Int, Tuple{Int, Int}}, Any}()
    for i in 1:2, j in 1:2
        data[(i, j, (0, 0))] = enc(A[i, j])
        for a in 1:2, b in 1:2
            data[(i, j, derivative_exponent_multiindex_2d(a, b))] =
                enc(N[(i, j, a, b)])
        end
    end

    derivatives = (
        deriv_num = Dict(key => enclosure_to_coeffs(value) for (key, value) in data),
        D = c(2),
        pack = (; data),
        pdeg = 1,
    )
    ξ = (big(2)//big(1), big(-1)//big(1))
    base = prepare_hsc_base_coeffs(derivatives; pdeg = 1)
    prepared = prepare_hsc_direction_coeffs(derivatives, base, ξ; pdeg = 1)
    B = PolytopeBox(1//4, 3//4, 1//4, 3//4)

    compact = compact_curvature_ranges_on_box(prepared, B)
    X = exact_big_interval(1//2)
    Y = exact_big_interval(1//2)
    geometry = evaluate_geometry_on_box(derivatives, X, Y)
    direct_Q = compute_Q_rigorously(geometry, ξ, X, Y)
    direct_R2 = riem_sq_range(geometry)

    @test compact !== nothing
    @test base.pdeg == 1
    @test size(prepared.S) == (1, 1)
    @test size(prepared.P) == (1, 1)
    @test size(prepared.R2_num) == (1, 1)
    @test interval_overlaps(compact.Q, direct_Q)
    @test interval_overlaps(compact.R2, direct_R2)
    @test sup(compact.Q) - inf(compact.Q) < big"1e-40"
    @test sup(compact.R2) - inf(compact.R2) < big"1e-40"
end
