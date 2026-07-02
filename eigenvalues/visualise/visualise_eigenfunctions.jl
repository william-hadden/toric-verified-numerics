using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "assemble_matrices.jl"))

midpoint(x) = Float64((inf(x) + sup(x)) / 2)

function read_mat4_variables(path)
    vars = Dict{String, Matrix{Float64}}()

    open(path, "r") do io
        while !eof(io)
            read(io, Int32)
            rows = read(io, Int32)
            cols = read(io, Int32)
            read(io, Int32)
            name_len = read(io, Int32)
            name = String(read(io, name_len - 1))
            read(io, UInt8)
            data = Vector{Float64}(undef, rows * cols)
            read!(io, data)
            vars[name] = reshape(data, rows, cols)
        end
    end

    return vars
end

function midpoint_entries(path)
    vars = read_mat4_variables(path)
    return (;
        n = round(Int, vars["n"][1]),
        i = round.(Int, vec(vars["i"])),
        j = round.(Int, vec(vars["j"])),
        values = (vec(vars["lo"]) .+ vec(vars["hi"])) ./ 2,
    )
end

function midpoint_eigenpairs(k = 8)
    Kdata = midpoint_entries(joinpath(@__DIR__, "..", "stiff_matrix.mat"))
    Mdata = midpoint_entries(joinpath(@__DIR__, "..", "mass_matrix.mat"))
    K = zeros(Float64, Kdata.n, Kdata.n)
    masses = zeros(Float64, Mdata.n)

    for a in eachindex(Kdata.values)
        K[Kdata.i[a], Kdata.j[a]] = Kdata.values[a]
    end
    for a in eachindex(Mdata.values)
        masses[Mdata.i[a]] = Mdata.values[a]
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

rotkey((a, b), k) = k == 0 ? (a, b) : rotkey((-b, a + b), k - 1)
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

function write_svg(path, xy, values, lambdas)
    width = 1050
    height = 700
    panel_w = width / 3
    panel_h = height / 2
    margin = 38
    xmin, xmax = extrema(xy[:, 1])
    ymin, ymax = extrema(xy[:, 2])
    span = max(xmax - xmin, ymax - ymin)

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")

        for j in 1:min(size(values, 2), 6)
            col = (j - 1) % 3
            row = (j - 1) ÷ 3
            ox = col * panel_w
            oy = row * panel_h
            scale = (min(panel_w, panel_h) - 2margin) / span
            vals = values[:, j]
            vmax = maximum(abs.(vals))

            println(io, @sprintf("""<text x="%.3f" y="%.3f" font-family="Helvetica, Arial, sans-serif" font-size="17" fill="#111">eig %d, lambda %.6g</text>""", ox + 18, oy + 25, j, lambdas[j]))

            for i in axes(xy, 1)
                x = ox + panel_w / 2 + scale * xy[i, 1]
                y = oy + panel_h / 2 - scale * xy[i, 2]
                fill = color_for(vals[i] / vmax)
                println(io, @sprintf("""<circle cx="%.4f" cy="%.4f" r="2.7" fill="%s" stroke="#222" stroke-width="0.2"/>""", x, y, fill))
            end
        end

        println(io, "</svg>")
    end
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

    svg_path = joinpath(@__DIR__, "eigenfunctions_midpoint_first6.svg")
    values = copy(Vn[:, 1:6])
    for j in axes(values, 2)
        if sum(data.masses .* values[:, j]) < 0
            values[:, j] .*= -1
        end
    end
    write_svg(svg_path, data.xy, values, lambdas)
    println()
    println("wrote ", svg_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
