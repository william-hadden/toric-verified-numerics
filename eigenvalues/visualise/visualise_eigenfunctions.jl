using JSON
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "assemble_matrices.jl"))

midpoint(x) = Float64((inf(x) + sup(x)) / 2)

function qdelta_triangulation(; delta = DEFAULT_DELTA, N::Integer = DEFAULT_N)
    pmesh = pdelta_mesh_data(; delta, N)
    I(x) = interval(BigFloat(x, RoundDown), BigFloat(x, RoundUp))
    vertex_ids = Dict{Tuple{Int, Int}, Int}()
    vertex_keys = Tuple{Int, Int}[]

    function vertex_id(key)
        if haskey(vertex_ids, key)
            return vertex_ids[key]
        end
        push!(vertex_keys, key)
        vertex_ids[key] = length(vertex_keys)
        return length(vertex_keys)
    end

    triangles = Tuple{Int, Int, Int}[]
    source_triangles = Int[]
    for k in 0:5, (t, tri) in pairs(pmesh.triangles)
        ids = ntuple(a -> vertex_id(rotkey(pmesh.node_keys[tri[a]], k)), 3)
        push!(triangles, ids)
        push!(source_triangles, t)
    end

    nodes = Matrix{Interval{BigFloat}}(undef, length(vertex_keys), 2)
    for (i, key) in pairs(vertex_keys)
        nodes[i, 1] = I(key[1] * pmesh.h)
        nodes[i, 2] = I(key[2] * pmesh.h)
    end

    return (; nodes, triangles, source_triangles, vertex_keys)
end

function assemble_qdelta_cr_matrices(nodes, triangles, metric_integrals; delta = DEFAULT_DELTA, N::Integer = DEFAULT_N)
    qmesh = qdelta_triangulation(; delta, N)
    edges, triangle_edges = build_cr_edges(qmesh.triangles)
    local_mass = [local_cr_mass_matrix(nodes, tri) for tri in triangles]
    local_stiffness = [
        local_cr_stiffness_matrix_from_integrals(nodes, triangles[t], metric_integrals[t])
        for t in eachindex(triangles)
    ]
    mass = Dict{Tuple{Int, Int}, Interval{BigFloat}}()
    stiffness = Dict{Tuple{Int, Int}, Interval{BigFloat}}()

    for t in eachindex(qmesh.triangles)
        source = qmesh.source_triangles[t]
        dofs = triangle_edges[t]
        addblock!(mass, dofs, local_mass[source])
        addblock!(stiffness, dofs, local_stiffness[source])
    end

    return (; matrix_size = length(edges), edges, mass, stiffness)
end

function midpoint_eigenpairs(k = 8)
    pmesh = pdelta_mesh_data()
    nodes, triangles = pmesh.nodes, pmesh.triangles
    println("assembling full midpoint eigenproblem for visualisation")
    metric_integrals = metric_integrals_on_pdelta_triangulation(nodes, triangles)
    assembly = assemble_qdelta_cr_matrices(nodes, triangles, metric_integrals)

    K = zeros(Float64, assembly.matrix_size, assembly.matrix_size)
    masses = zeros(Float64, assembly.matrix_size)

    for ((i, j), v) in assembly.stiffness
        K[i, j] = midpoint(v)
    end
    for ((i, _), v) in assembly.mass
        masses[i] = midpoint(v)
    end

    invsqrt = 1 ./ sqrt.(masses)
    A = Symmetric((K .+ transpose(K)) .* ((invsqrt * transpose(invsqrt)) ./ 2))
    E = eigen(A)
    p = sortperm(E.values)[1:k]
    return E.values[p], E.vectors[:, p] .* reshape(invsqrt, :, 1), masses
end

function edge_data(masses)
    qmesh = qdelta_triangulation()
    edges, _ = build_cr_edges(qmesh.triangles)

    xy = zeros(Float64, length(edges), 2)
    keysum = Vector{Tuple{Int, Int}}(undef, length(edges))

    for (e, (i, j)) in pairs(edges)
        xy[e, 1] = (midpoint(qmesh.nodes[i, 1]) + midpoint(qmesh.nodes[j, 1])) / 2
        xy[e, 2] = (midpoint(qmesh.nodes[i, 2]) + midpoint(qmesh.nodes[j, 2])) / 2

        ki = qmesh.vertex_keys[i]
        kj = qmesh.vertex_keys[j]
        keysum[e] = (ki[1] + kj[1], ki[2] + kj[2])
    end

    return (; xy, keysum, masses)
end

minner(masses, a, b) = sum(masses .* a .* b)
mnorm(masses, a) = sqrt(max(minner(masses, a, a), 0.0))
normalize_m(masses, v) = v ./ mnorm(masses, v)

function weighted_q(masses, A)
    Q = Matrix(qr(sqrt.(masses) .* A).Q)
    return Q[:, 1:size(A, 2)]
end

reflkey((a, b)) = (b, a)

function symmetry_permutations(keysum)
    edge_id = Dict(keysum[i] => i for i in eachindex(keysum))
    rotperm = [edge_id[rotkey(keysum[i], 1)] for i in eachindex(keysum)]
    reflperm = [edge_id[reflkey(keysum[i])] for i in eachindex(keysum)]
    return rotperm, reflperm
end

function color_for(t)
    t = clamp(t, -1.0, 1.0)
    if t < 0
        a = t + 1
        r = round(Int, 255 * a)
        g = round(Int, 255 * a)
        b = 255
    else
        a = 1 - t
        r = 255
        g = round(Int, 255 * a)
        b = round(Int, 255 * a)
    end
    return @sprintf("#%02x%02x%02x", r, g, b)
end

function render_png(path, width, height, items)
    spec_path = tempname() * ".json"
    open(spec_path, "w") do io
        JSON.print(io, Dict(
            "output" => path,
            "width" => width,
            "height" => height,
            "scale" => 3,
            "items" => items,
        ))
    end
    try
        run(`python3 $(joinpath(@__DIR__, "render_png.py")) $spec_path`)
    finally
        rm(spec_path; force = true)
    end
end

function write_png(path, xy, values, lambdas)
    width = 1050
    height = 700
    panel_w = width / 3
    panel_h = height / 2
    margin = 38
    xmin, xmax = extrema(xy[:, 1])
    ymin, ymax = extrema(xy[:, 2])
    span = max(xmax - xmin, ymax - ymin)
    items = Any[]

    for j in 1:min(size(values, 2), 6)
        col = (j - 1) % 3
        row = (j - 1) ÷ 3
        ox = col * panel_w
        oy = row * panel_h
        scale = (min(panel_w, panel_h) - 2margin) / span
        vals = values[:, j]
        vmax = maximum(abs.(vals))

        push!(items, Dict(
            "type" => "text",
            "position" => [ox + 18, oy + 12],
            "text" => @sprintf("eig %d, lambda %.6g", j, lambdas[j]),
            "size" => 17,
            "fill" => "#111111",
        ))

        for i in axes(xy, 1)
            x = ox + panel_w / 2 + scale * xy[i, 1]
            y = oy + panel_h / 2 - scale * xy[i, 2]
            fill = color_for(vals[i] / vmax)
            push!(items, Dict(
                "type" => "circle",
                "center" => [x, y],
                "radius" => 2.7,
                "fill" => fill,
                "outline" => "#222222",
                "width" => 0.2,
            ))
        end
    end

    render_png(path, width, height, items)
end

function main()
    lambdas, V, masses = midpoint_eigenpairs()
    data = edge_data(masses)

    Vn = hcat([normalize_m(data.masses, V[:, i]) for i in axes(V, 2)]...)
    constv = ones(size(V, 1))
    x = data.xy[:, 1]
    y = data.xy[:, 2]

    Qeig23 = weighted_q(data.masses, V[:, 2:3])
    Qlin = weighted_q(data.masses, hcat(x, y))
    principal_cosines = svdvals(Qeig23' * Qlin)
    relative_linear_residual = norm((I - Qeig23 * Qeig23') * Qlin) / norm(Qlin)

    rotperm, reflperm = symmetry_permutations(data.keysum)

    println("First 8 midpoint eigenvalues:")
    for i in 1:8
        @printf("%d %.15g\n", i, lambdas[i])
    end

    println()
    println("M-weighted constant correlation for eig 1:")
    println(abs(minner(data.masses, Vn[:, 1], constv) / mnorm(data.masses, constv)))

    println()
    println("Principal cosines between span(eig 2, eig 3) and span(x, y):")
    println(principal_cosines)
    println("Relative residual of linear span after projection to eig 2/3 span:")
    println(relative_linear_residual)

    println()
    println("Individual D6 symmetry residuals:")
    for j in 1:8
        v = Vn[:, j]
        rot_res = mnorm(data.masses, v[rotperm] - v) / mnorm(data.masses, v)
        refl_res = mnorm(data.masses, v[reflperm] - v) / mnorm(data.masses, v)
        @printf("eig %d: rotation %.3e, reflection %.3e\n", j, rot_res, refl_res)
    end

    png_path = joinpath(@__DIR__, "eigenfunctions_midpoint_first6.png")
    values = copy(Vn[:, 1:6])
    for j in axes(values, 2)
        if sum(data.masses .* values[:, j]) < 0
            values[:, j] .*= -1
        end
    end
    write_png(png_path, data.xy, values, lambdas)
    println()
    println("wrote ", png_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
