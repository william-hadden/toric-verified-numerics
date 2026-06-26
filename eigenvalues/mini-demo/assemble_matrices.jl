using DelaunayTriangulation
using IntervalArithmetic

N = 16
dir = @__DIR__

xs = Float64[]
ys = Float64[]
for j = 0:N, i = 0:N
    push!(xs, i/N); push!(ys, j/N)
end
for j = 0:N-1, i = 0:N-1
    push!(xs, (i+0.5)/N); push!(ys, (j+0.5)/N)
end

nodes = [xs ys]
T = collect(each_solid_triangle(triangulate(transpose(nodes); randomise=false)))
P = interval.(nodes)
n = size(P, 1)
K = Dict{Tuple{Int,Int},Interval{Float64}}() # stiffness matrix
M = Dict{Tuple{Int,Int},Interval{Float64}}() # mass matrix

function addblock!(A, verts, E)
    for a = 1:3, b = 1:3
        key = (verts[a], verts[b])
        A[key] = get(A, key, interval(0.0)) + E[a,b]
    end
end

for tri in T
    verts = collect(tri)
    x1, y1 = P[verts[1],1], P[verts[1],2]
    x2, y2 = P[verts[2],1], P[verts[2],2]
    x3, y3 = P[verts[3],1], P[verts[3],2]

    detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)

    # ensure triangles are positively oriented
    if sup(detJ) < 0
        verts = verts[[1,3,2]]
        x1, y1 = P[verts[1],1], P[verts[1],2]
        x2, y2 = P[verts[2],1], P[verts[2],2]
        x3, y3 = P[verts[3],1], P[verts[3],2]
        detJ = -detJ
    end
    @assert inf(detJ) > 0 # this catches the case where detJ is an interval that contains 0 (-> error)

    area = detJ / interval(2.0)
    G = [y2-y3 y3-y1 y1-y2; x3-x2 x1-x3 x2-x1] / (interval(2.0)*area)

    addblock!(K, verts, area * (G' * G))
    addblock!(M, verts, (area/interval(12.0)) * interval.([2.0 1.0 1.0; 1.0 2.0 1.0; 1.0 1.0 2.0]))
end

function mat4write(path, entries)
    open(path, "w") do io
        for (name, value) in entries
            A = value isa Number ? reshape([Float64(value)], 1, 1) : reshape(Float64.(value), :, 1)
            write(io, Int32(0), Int32(size(A,1)), Int32(size(A,2)), Int32(0), Int32(length(name)+1))
            write(io, codeunits(name)); write(io, UInt8(0)); write(io, vec(A))
        end
    end
end

function savematrix(path, A)
    ij = sort!(collect(keys(A)))
    mat4write(path, [
        "n" => n,
        "i" => first.(ij),
        "j" => last.(ij),
        "lo" => [inf(A[key]) for key in ij],
        "hi" => [sup(A[key]) for key in ij],
    ])
end

savematrix(joinpath(dir, "stiff_matrix.mat"), K)
savematrix(joinpath(dir, "mass_matrix.mat"), M)
