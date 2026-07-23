"""
Read the MATLAB version-4 file at `path`.

Return its numeric variables as a dictionary of `Float64` matrices.
"""
function read_mat4_variables(path)
    variables = Dict{String,Matrix{Float64}}()
    open(path, "r") do io
        while !eof(io)
            read(io, Int32)
            rows = read(io, Int32)
            columns = read(io, Int32)
            read(io, Int32)
            name_length = read(io, Int32)
            name = String(read(io, name_length - 1))
            read(io, UInt8)
            values = Vector{Float64}(undef, rows * columns)
            read!(io, values)
            variables[name] = reshape(values, rows, columns)
        end
    end
    return variables
end
