using JSON
using LinearAlgebra
using Printf
using SparseArrays

include(joinpath(@__DIR__, "..", "assemble_matrices.jl"))

"""Rotate an integer lattice key by `steps` sixth-turns."""
function rotate_key(key, steps::Integer)
    for _ in 1:steps
        key = (-key[2], key[1] + key[2])
    end
    return key
end

"""Read all real arrays from a MATLAB version-4 binary file."""
function read_mat4_variables(path)
    variables = Dict{String,Matrix{Float64}}()
    open(path, "r") do io
        while !eof(io)
            read(io, Int32)
            rows = Int(read(io, Int32))
            columns = Int(read(io, Int32))
            read(io, Int32)
            name_length = Int(read(io, Int32))
            name = String(read(io, name_length - 1))
            read(io, UInt8)
            data = Vector{Float64}(undef, rows * columns)
            read!(io, data)
            variables[name] = reshape(data, rows, columns)
        end
    end
    return variables
end

"""Return the midpoint sparse matrix stored as interval endpoint arrays."""
function midpoint_sparse_matrix(path)
    data = read_mat4_variables(path)
    required = ("n", "i", "j", "lo", "hi")
    all(name -> haskey(data, name), required) || error("Incomplete matrix file $path")
    n = round(Int, data["n"][1])
    rows = round.(Int, vec(data["i"]))
    columns = round.(Int, vec(data["j"]))
    values = (vec(data["lo"]) .+ vec(data["hi"])) ./ 2
    return sparse(rows, columns, values, n, n)
end

"""Remove the mass mean and normalize a coefficient vector in place."""
function center_and_normalize!(vector, mass)
    vector .-= dot(mass, vector) / sum(mass)
    norm_squared = dot(vector, mass .* vector)
    norm_squared > 0 || error("Cannot normalize a constant FEM vector")
    vector ./= sqrt(norm_squared)
    return vector
end

"""Return a deterministic D6-invariant radial vector on quotient edges."""
function quotient_radius_vector(mesh, edges, quotient, count)
    radii = zeros(Float64, count)
    assigned = falses(count)
    for (edge_index, (left, right)) in pairs(edges)
        a = mesh.node_keys[left][1] + mesh.node_keys[right][1]
        b = mesh.node_keys[left][2] + mesh.node_keys[right][2]
        radius = Float64(a^2 + a * b + b^2)
        index = quotient[edge_index]
        assigned[index] && radii[index] != radius &&
            error("A quotient class contains unequal D6 radii")
        radii[index] = radius
        assigned[index] = true
    end
    all(assigned) || error("Some quotient degrees of freedom have no edge")
    return radii
end

"""Compute the first nonconstant midpoint eigenpair by inverse iteration."""
function inverse_iteration(stiffness, mass_matrix, mass, start)
    vector = center_and_normalize!(copy(start), mass)
    factor = cholesky(Symmetric(stiffness + mass_matrix))
    for _ in 1:50
        next = factor \ (mass .* vector)
        center_and_normalize!(next, mass)
        dot(next, mass .* vector) < 0 && (next .*= -1)
        vector = next
    end
    stiff_vector = stiffness * vector
    lambda = dot(vector, stiff_vector) / dot(vector, mass .* vector)
    residual = norm(stiff_vector - lambda * (mass .* vector)) / norm(stiff_vector)
    residual < 1e-9 || error("Midpoint inverse iteration did not converge")
    return (; lambda, vector, residual)
end

"""Load the midpoint problem and recover its first nonconstant eigenpair."""
function first_nonconstant_eigenpair(mesh, edges, quotient)
    stiffness = midpoint_sparse_matrix(joinpath(@__DIR__, "..", "stiff_matrix.mat"))
    mass_matrix = midpoint_sparse_matrix(joinpath(@__DIR__, "..", "mass_matrix.mat"))
    size(stiffness) == size(mass_matrix) || error("The FEM matrix sizes disagree")
    maximum(quotient) == size(stiffness, 1) || error("The mesh and FEM matrices disagree")
    mass = Vector(diag(mass_matrix))
    start = quotient_radius_vector(mesh, edges, quotient, length(mass))
    center_and_normalize!(start, mass)
    pair = inverse_iteration(stiffness, mass_matrix, mass, start)
    dot(pair.vector, mass .* start) < 0 && (pair.vector .*= -1)
    verified = read_mat4_variables(joinpath(@__DIR__, "..", "verified_eigenvalue.mat"))
    lower = verified["lambda_fem_lb"][1]
    upper = verified["lambda_fem_ub"][1]
    lower <= pair.lambda <= upper || error("The midpoint eigenvalue is not certified")
    return pair
end

"""Convert one exact mesh key to a floating-point plotting coordinate."""
function plot_point(mesh, key)
    factor = mesh.h / mesh.scale
    return (Float64(key[1] * factor), Float64(key[2] * factor))
end

"""Return the coordinate extent of the requested rotational copies."""
function domain_extent(mesh, rotations)
    xmin, xmax = Inf, -Inf
    ymin, ymax = Inf, -Inf
    for rotation in rotations, key in mesh.node_keys
        x, y = plot_point(mesh, rotate_key(key, rotation))
        xmin, xmax = min(xmin, x), max(xmax, x)
        ymin, ymax = min(ymin, y), max(ymax, y)
    end
    return xmin, xmax, ymin, ymax
end

"""Construct an aspect-preserving map into a rectangular image panel."""
function screen_map(mesh, rotations, panel; margin = 20)
    xmin, xmax, ymin, ymax = domain_extent(mesh, rotations)
    available_width = panel.width - 2margin
    available_height = panel.height - 2margin
    scale = min(available_width / (xmax - xmin), available_height / (ymax - ymin))
    used_width = scale * (xmax - xmin)
    used_height = scale * (ymax - ymin)
    xoffset = panel.x + (panel.width - used_width) / 2
    yoffset = panel.y + (panel.height - used_height) / 2
    return function ((x, y))
        return (xoffset + scale * (x - xmin),
                yoffset + used_height - scale * (y - ymin))
    end
end

"""Return screen coordinates for one rotated mesh triangle."""
function screen_triangle(mesh, triangle, rotation, project)
    return [
        collect(project(plot_point(mesh, rotate_key(mesh.node_keys[index], rotation))))
        for index in triangle
    ]
end

"""Create the polygon items for a triangulation and its rotational copies."""
function triangulation_items(mesh, rotations, project)
    colors = ["#dceeff", "#ffe7c7", "#dff3dc", "#f4d9ff", "#ffe0e5", "#d8f0ef"]
    items = Any[]
    for rotation in rotations, triangle in mesh.triangles
        push!(items, Dict(
            "type" => "polygon",
            "points" => screen_triangle(mesh, triangle, rotation, project),
            "fill" => colors[mod(rotation, length(colors)) + 1],
            "outline" => "#87929b",
            "width" => 0.25,
        ))
    end
    return items
end

"""Return the triangle-local vertex traces of a CR finite element function."""
function local_vertex_values(vector, quotient, edge_ids)
    u1, u2, u3 = (vector[quotient[index]] for index in edge_ids)
    return (
        -u1 + u2 + u3,
        u1 - u2 + u3,
        u1 + u2 - u3,
    )
end

"""Map a signed normalized value to a blue-white-red color."""
function field_color(value)
    blue, white, red = (33, 102, 172), (247, 247, 247), (178, 24, 43)
    t = clamp(value, -1.0, 1.0)
    left, right, fraction = t < 0 ? (blue, white, t + 1) : (white, red, t)
    channel(index) = round(Int, (1 - fraction) * left[index] + fraction * right[index])
    return @sprintf("#%02x%02x%02x", channel(1), channel(2), channel(3))
end

"""Return local CR vertex values and their common symmetric color limit."""
function local_field_data(vector, quotient, triangle_edge_ids)
    values = [
        local_vertex_values(vector, quotient, edge_ids)
        for edge_ids in triangle_edge_ids
    ]
    limit = maximum(abs(value) for local_values in values for value in local_values)
    limit > 0 || error("The plotted eigenfunction is zero")
    return values, limit
end

"""Create an affine CR field item on the requested rotational copies."""
function field_item(mesh, rotations, project, local_values, limit)
    triangles = Any[]
    values = Any[]
    for rotation in rotations, index in eachindex(mesh.triangles)
        push!(triangles, screen_triangle(
            mesh,
            mesh.triangles[index],
            rotation,
            project,
        ))
        push!(values, collect(local_values[index]))
    end
    return Dict(
        "type" => "field",
        "triangles" => triangles,
        "values" => values,
        "limit" => limit,
        "maximum_leaf_diameter" => 4,
    )
end

"""Return the screen-space boundary of the triangle or inset hexagon."""
function boundary_item(mesh, rotations, project)
    lattice_radius = maximum(first, mesh.node_keys)
    seed = (lattice_radius, 0)
    keys = length(rotations) == 1 ? [(0, 0), seed, (lattice_radius, -lattice_radius)] :
        [rotate_key(seed, rotation) for rotation in rotations]
    points = [collect(project(plot_point(mesh, key))) for key in keys]
    push!(points, first(points))
    return Dict("type" => "line", "points" => points, "fill" => "#20262b", "width" => 1.2)
end

"""Create a compact diverging color bar for a scalar field."""
function colorbar_items(x, y, width, height, limit)
    items = Any[]
    steps = 80
    for index in 0:(steps - 1)
        left = x + width * index / steps
        right = x + width * (index + 1) / steps
        value = -1 + 2 * (index + 0.5) / steps
        push!(items, Dict(
            "type" => "polygon",
            "points" => [[left, y], [right, y], [right, y + height], [left, y + height]],
            "fill" => field_color(value),
        ))
    end
    labels = ((x, -limit), (x + width / 2, 0.0), (x + width, limit))
    for (position, value) in labels
        push!(items, Dict("type" => "text", "position" => [position - 18, y + height + 5],
                          "text" => @sprintf("%.2f", value), "size" => 15, "fill" => "#20262b"))
    end
    return items
end

"""Render a JSON drawing specification through the local PNG helper."""
function render_png(path, width, height, items)
    mktemp(@__DIR__) do specification_path, io
        JSON.print(io, Dict(
            "output" => path,
            "width" => width,
            "height" => height,
            "scale" => 3,
            "items" => items,
        ))
        close(io)
        run(`python3 $(joinpath(@__DIR__, "render_png.py")) $specification_path`)
    end
    return path
end

"""Write the fundamental-triangle and inset-hexagon triangulation images."""
function write_triangulation_images(mesh)
    triangle_path = joinpath(@__DIR__, "triangle_triangulation.png")
    hexagon_path = joinpath(@__DIR__, "hexagon_triangulation.png")
    triangle_panel = (x = 20, y = 65, width = 1160, height = 855)
    hexagon_panel = (x = 20, y = 65, width = 1260, height = 905)
    triangle_map = screen_map(mesh, 0:0, triangle_panel; margin = 15)
    hexagon_map = screen_map(mesh, 0:5, hexagon_panel; margin = 15)
    triangle_items = triangulation_items(mesh, 0:0, triangle_map)
    hexagon_items = triangulation_items(mesh, 0:5, hexagon_map)
    push!(triangle_items, boundary_item(mesh, 0:0, triangle_map))
    push!(hexagon_items, boundary_item(mesh, 0:5, hexagon_map))
    pushfirst!(triangle_items, Dict("type" => "text", "position" => [28, 18],
        "text" => "Triangulation of the fundamental triangle", "size" => 24, "fill" => "#20262b"))
    pushfirst!(hexagon_items, Dict("type" => "text", "position" => [28, 18],
        "text" => "Triangulation of the inset hexagon", "size" => 24, "fill" => "#20262b"))
    render_png(triangle_path, 1200, 940, triangle_items)
    render_png(hexagon_path, 1300, 990, hexagon_items)
    return triangle_path, hexagon_path
end

"""Write the two-panel first nonconstant FEM eigenfunction image."""
function write_eigenfunction_image(mesh, triangle_edge_ids, quotient, pair)
    path = joinpath(@__DIR__, "first_nonconstant_fem_eigenfunction.png")
    triangle_panel = (x = 25, y = 115, width = 700, height = 535)
    hexagon_panel = (x = 755, y = 115, width = 920, height = 535)
    triangle_map = screen_map(mesh, 0:0, triangle_panel; margin = 12)
    hexagon_map = screen_map(mesh, 0:5, hexagon_panel; margin = 12)
    local_values, limit = local_field_data(pair.vector, quotient, triangle_edge_ids)
    items = Any[
        field_item(mesh, 0:0, triangle_map, local_values, limit),
        field_item(mesh, 0:5, hexagon_map, local_values, limit),
    ]
    push!(items, boundary_item(mesh, 0:0, triangle_map))
    push!(items, boundary_item(mesh, 0:5, hexagon_map))
    push!(items, Dict("type" => "text", "position" => [28, 18],
        "text" => "First non-constant D6-invariant FEM eigenfunction",
        "size" => 25, "fill" => "#20262b"))
    push!(items, Dict("type" => "text", "position" => [28, 53],
        "text" => @sprintf("Midpoint visualisation, lambda_FEM = %.12f", pair.lambda),
        "size" => 17, "fill" => "#53606b"))
    push!(items, Dict("type" => "text", "position" => [255, 88],
        "text" => "Fundamental triangle", "size" => 19, "fill" => "#20262b"))
    push!(items, Dict("type" => "text", "position" => [1115, 88],
        "text" => "Inset hexagon", "size" => 19, "fill" => "#20262b"))
    append!(items, colorbar_items(650, 680, 400, 18, limit))
    render_png(path, 1700, 750, items)
    return path
end

"""Regenerate the three visualizations for the fixed production FEM problem."""
function main()
    setprecision(BigFloat, 256) do
        # These are the fixed production choices in ../orchestrate.jl.
        mesh = sector_mesh(1 // 5000, 20, 3, 1, 2, 1)
        edges, triangle_edge_ids = build_cr_edges(mesh.triangles)
        quotient = d6_edge_quotient(edges, mesh.node_keys)
        pair = first_nonconstant_eigenpair(mesh, edges, quotient)
        triangle_path, hexagon_path = write_triangulation_images(mesh)
        eigenfunction_path = write_eigenfunction_image(
            mesh,
            triangle_edge_ids,
            quotient,
            pair,
        )
        println("midpoint FEM eigenvalue: $(pair.lambda)")
        println("relative eigenvector residual: $(pair.residual)")
        println("wrote $triangle_path")
        println("wrote $hexagon_path")
        println("wrote $eigenfunction_path")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
