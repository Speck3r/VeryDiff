import Base: size

function size(Z::Zonotope, d::Integer)
    @assert d<=2
    return size(Z.G,d)
end

function truncate_network(T::Type,N::Network)
    layers = []
    for l in N.layers
        if l isa ReLU
            push!(layers, l)
        elseif l isa Dense
            push!(layers,Dense(convert.(Float64,convert.(T,deepcopy(l.W))),convert.(Float64,convert.(T,deepcopy(l.b)))))
        else
            throw("Unknown layer type")
        end
    end
    return Network(layers)
end

function to_diff_zono(task :: VerificationTask)
    input_dim = size(task.middle,1)
    if NEW_HEURISTIC
        Z1 = Zonotope(Matrix(I, input_dim, input_dim)[:,task.distance_indices] .* task.distance', task.middle, Matrix(1.0I, size(task.distance_indices,1), size(task.distance_indices,1)))
    else
        Z1 = Zonotope(Matrix(I, input_dim, input_dim)[:,task.distance_indices] .* task.distance', task.middle, nothing)
    end
    Z2 = deepcopy(Z1)
    return DiffZonotope(Z1, Z2, task.∂Z, 0, 0, 0)
end


"""
Test if univariate polynomial represented by coefficients [p₀, p₁, ...]
represents a linear function.

args:
    v - vector of polynomial coefficients

returns:
    true ⟺ polynomial represents linear function
"""
islinear(v) = length(v) < 3 ? true : all(v[3:end] .== 0)


"""
Given lower and upper bounds for x ∈ [l, u], returns masks 

neg = true iff always x ≤ 0
pos = true iff always x ≥ 0
unstable = true iff l < 0 < u
"""
function stability_mask(l, u)
    check = falses(length(l))
    neg = (u .<= 0.) .&& .!check
    check .|= neg
    pos = (l .>= 0) .&& .!check 
    check .|= pos 
    unstable = (l .< 0) .&& (u .> 0) .&& .!check 
    check .|= unstable
    @assert all(check) "Not all cases covered!"

    return neg, pos, unstable
end