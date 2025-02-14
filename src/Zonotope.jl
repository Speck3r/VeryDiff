# direction: 1 (maximize) or -1 (minimize)
function zono_optimize(direction::N, Z::Zonotope, d :: Int) where N<:Number
    @assert isone(direction) || isone(-direction)
    row = view(Z.G,d,:)
    result = direction*sum(abs,row) + Z.c[d]
    return result
end

function zono_bounds(Z::Zonotope)
    #return @timeit to "Zonotope_Bounds" begin
    b = sum(abs,Z.G;dims=2)
    #b = abs.(Z.G)*ones(size(Z.G,2))
    return [Z.c.-b b.+Z.c]
    #end
end

function zono_get_max_vector(Z::Zonotope, direction::AbstractVector{N}) where N<:Number
    weights = direction' * Z.G
    return -1.0*(weights .< 0.0) + 1.0*(weights .>= 0.0)
end

function zono_get_max_vector(Z::Zonotope, d)
    weights = Z.G[d,:]
    return -1.0*(weights .< 0.0) + 1.0*(weights .>= 0.0)
end


"""
Returns a random point contained in the zonotope.
"""
function random_point(z::Zonotope)
    n_generators = size(z.G, 2)
    ϵ = 2 .* rand(n_generators) .- 1
    x = z.G * ϵ .+ z.c
end
