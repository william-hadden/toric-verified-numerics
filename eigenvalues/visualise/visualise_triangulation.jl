using IntervalArithmetic
using Printf

include(joinpath(@__DIR__, "..", "assemble_matrices.jl"))

midpoint(x) = Float64((inf(x) + sup(x)) / 2)
node_point(nodes, i) = (midpoint(nodes[i, 1]), midpoint(nodes[i, 2]))

rotate60((x, y), k) = k == 0 ? (x, y) : rotate60((-y, x + y), k - 1)

function triangle_points(nodes, triangles; rotations = 0:0)
    out = []
    for k in rotations, tri in triangles
        push!(out, (sector = k, points = [rotate60(node_point(nodes, i), k) for i in tri]))
    end
    return out
end

function plot_box(tris)
    xs = [p[1] for tri in tris for p in tri.points]
    ys = [p[2] for tri in tris for p in tri.points]
    return minimum(xs), maximum(xs), minimum(ys), maximum(ys)
end

function screen_map(tris, width, height, margin)
    xmin, xmax, ymin, ymax = plot_box(tris)
    scale = min((width - 2margin) / (xmax - xmin), (height - 2margin) / (ymax - ymin))
    used_width = scale * (xmax - xmin)
    used_height = scale * (ymax - ymin)
    xoff = (width - used_width) / 2
    yoff = (height - used_height) / 2

    return function ((x, y))
        sx = xoff + scale * (x - xmin)
        sy = height - (yoff + scale * (y - ymin))
        return sx, sy
    end
end

function svg_points(points, project)
    return join((@sprintf("%.6f,%.6f", project(p)...) for p in points), " ")
end

function write_svg(path, tris; width = 900, height = 760, title = "")
    project = screen_map(tris, width, height, 40)
    colors = ["#dceeff", "#ffe7c7", "#dff3dc", "#f4d9ff", "#ffe0e5", "#d8f0ef"]

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        if !isempty(title)
            println(io, """<text x="24" y="34" font-family="Helvetica, Arial, sans-serif" font-size="22" fill="#222">$title</text>""")
        end

        for tri in tris
            fill = colors[mod(tri.sector, length(colors)) + 1]
            pts = svg_points(tri.points, project)
            println(io, """<polygon points="$pts" fill="$fill" fill-opacity="0.55" stroke="#222" stroke-width="1.2"/>""")
        end

        seen = Set{Tuple{Float64, Float64}}()
        for tri in tris, p in tri.points
            q = project(p)
            key = (round(q[1]; digits = 8), round(q[2]; digits = 8))
            key in seen && continue
            push!(seen, key)
            println(io, @sprintf("""<circle cx="%.6f" cy="%.6f" r="2.3" fill="#111"/>""", q[1], q[2]))
        end

        println(io, "</svg>")
    end
end

function write_triangulation_plots(nodes, triangles; outdir = @__DIR__)
    p_tris = triangle_points(nodes, triangles)
    q_tris = triangle_points(nodes, triangles; rotations = 0:5)

    p_path = joinpath(outdir, "pdelta_triangulation.svg")
    q_path = joinpath(outdir, "qdelta_triangulation.svg")

    write_svg(p_path, p_tris; title = "P_delta triangulation")
    write_svg(q_path, q_tris; title = "Q_delta triangulation")

    return (; p_path, q_path)
end

function main()
    nodes, triangles = get_pdelta_triangulation()
    paths = write_triangulation_plots(nodes, triangles)
    println("wrote ", paths.p_path)
    println("wrote ", paths.q_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
