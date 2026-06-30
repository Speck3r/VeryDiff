
"""
    `propagate_layer!(ZoutRefVec :: Vector{Zonotope}, L :: ONNXLinear{S1}, inputs :: Vector{Zonotope}) where {S1}`

    Propagates the input zonotopes through a linear layer and updates the output zonotopes in-place with the result. 

    args:
    - `ZoutRefVec`: A vector of output zonotope references to be updated. For a linear layer, this should contain exactly one zonotope reference.
    - `L`: The linear layer 
    - `inputs`: A vector of input zonotopes. For a linear layer, this should contain exactly one zonotope.
"""
function propagate_layer!(ZoutRefVec :: Vector{Zonotope}, L :: ONNXLinear{S1}, inputs :: Vector{Zonotope}) where {S1}
    @assert length(inputs) == 1 "Dense layer should have exactly one input"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]
    return propagate_layer!(ZoutRef, L, Zin)
end

function propagate_layer!(ZoutRef :: Zonotope, L :: ONNXLinear{S1}, Zin :: Zonotope) where {S1}
    # Zout must have exactly the same ids as Zin
    # @assert all(ZoutRef.generator_ids .== Zin.generator_ids) "Zonotope generator IDs do not match during Dense propagation!"
    for i in 1:length(Zin.Gs)
        mul!(ZoutRef.Gs[i], L.dense.weight, Zin.Gs[i])
    end
    mul!(ZoutRef.c, L.dense.weight, Zin.c)
    ZoutRef.c .+= L.dense.bias
end

function propagate_layer!(ZoutRef :: Zonotope, L :: ONNXAddConst{S1}, Zin :: Zonotope) where {S1}
    # Zout must have exactly the same ids as Zin
    # @assert all(ZoutRef.generator_ids .== Zin.generator_ids) "Zonotope generator IDs do not match during Dense propagation!"
    for i in 1:length(Zin.Gs)
        ZoutRef.Gs[i] .= Zin.Gs[i]
    end
    ZoutRef.c .= Zin.c .+ L.c
end

function get_slope(l,u, alpha)
    if u <= 0
        return 0.0
    elseif l >= 0
        return 1.0
    else
        return alpha
    end
end

function propagate_layer!(ZoutRefVec :: Vector{Zonotope}, L :: ONNXRelu{S}, inputs :: Vector{Zonotope}; lower=nothing, upper=nothing) where {S}
    @assert length(inputs) == 1 "Dense layer should have exactly one input"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]
    return propagate_layer!(ZoutRef, L, Zin; lower=lower, upper=upper)
end

function propagate_layer!(ZoutRef :: Zonotope, _L :: ONNXRelu{S}, Zin :: Zonotope; lower=nothing, upper=nothing) where {S}
    if isnothing(lower) || isnothing(upper)
        bounds = zono_bounds(Zin)
        lower = @view bounds[:,1]
        upper = @view bounds[:,2]
    end

    dim = length(lower)
    crossing = @simd_bool_expr dim ((lower < 0.0) & (upper > 0.0))
    α = clamp.(upper./(upper.-lower),0.0,1.0)
    # Use is_onesided to compute 
    λ = ifelse.(crossing, α, ifelse.(lower .>= 0.0, 1.0, 0.0))

    new_gens = count(crossing)
    
    γ = 0.5 .* max.(-λ .* lower,0.0,((-).(1.0,λ)).*upper)  # Computed offset (-λl/2)

    ZoutRef.c .= λ .* Zin.c .+ crossing.*γ

    indices = intersect_indices(ZoutRef.generator_ids, Zin.generator_ids)
    if VeryDiff.NEW_HEURISTIC[]
        influence_new = ZoutRef.influence
        column_pos = size(influence_new[ZoutRef.owned_generators],2) - new_gens + 1
        # @debug "Adding $new_gens new columns at position $column_pos to influence matrix of owned generator ID $(ZoutRef.generator_ids[ZoutRef.owned_generators])"
        # @debug "Sizes of influence matrices: $([size(inf) for inf in Zin.influence])"
        # Other influence matrices remain the same
        # Only need to update the owned generator influence matrix
        if !isnothing(Zin.owned_generators) && Zin.owned_generators == attempt_find_index_position(Zin.generator_ids, ZoutRef.generator_ids[ZoutRef.owned_generators])
            influence_new[ZoutRef.owned_generators][:, 1:column_pos-1] .= Zin.influence[Zin.owned_generators]
        end
        # @debug "Size of owned influence matrix after copy: $(size(influence_new[ZoutRef.owned_generators]))"
        influence_new[ZoutRef.owned_generators][:,column_pos:end] .= 0.0
        bounds_range = upper[crossing] .- lower[crossing]
        @inbounds for (idx, g) in enumerate(Zin.Gs)
            influence_new[ZoutRef.owned_generators][:,column_pos:end] .+= Zin.influence[idx] * abs.((@view g[crossing,:]) ./ bounds_range)'
        end
    else
        influence_new = Zin.influence
    end

    num_new_gens = count(crossing)

    updateGeneratorsMul!(ZoutRef.Gs, indices, Zin.Gs, λ, :)
    ZoutRef.Gs[ZoutRef.owned_generators][:,(end-num_new_gens+1):end] .= 0.0
    generator_offset = size(ZoutRef.Gs[ZoutRef.owned_generators],2) - num_new_gens
    A = ZoutRef.Gs[ZoutRef.owned_generators]
    @inbounds for (i, row) in enumerate(findall(crossing))
        A[row, (generator_offset + i)] = abs(γ[row])
    end
end


function propagate_layer!(ZoutRefVec :: Vector{Zonotope}, L :: ONNXPoly{S}, inputs :: Vector{Zonotope}; lower=nothing, upper=nothing) where {S}
    @assert length(inputs) == 1 "Poly layer should have exactly one input"
    @assert length(ZoutRefVec) == 1 "Poly layer should have exactly one output"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]
    return propagate_layer!(ZoutRef, L, Zin; lower=lower, upper=upper)
end

function propagate_layer!(ZoutRef :: Zonotope, L :: Union{ONNXPoly{S}, ONNXGelu{S}}, Zin :: Zonotope; lower=nothing, upper=nothing) where {S}
    # TODO: use bounds stored in the layer for tightening
    if isnothing(lower) || isnothing(upper)
        bounds = zono_bounds(Zin)
        lower = @view bounds[:,1]
        upper = @view bounds[:,2]
    end

    λ, β, γ = get_linear_relaxation(L, lower, upper)

    ZoutRef.c .= λ .* Zin.c .+ β

    new_gens = size(lower, 1)

    indices = intersect_indices(ZoutRef.generator_ids, Zin.generator_ids)
    if VeryDiff.NEW_HEURISTIC[]
        influence_new = ZoutRef.influence
        column_pos = size(influence_new[ZoutRef.owned_generators],2) - new_gens + 1
        # @debug "Adding $new_gens new columns at position $column_pos to influence matrix of owned generator ID $(ZoutRef.generator_ids[ZoutRef.owned_generators])"
        # @debug "Sizes of influence matrices: $([size(inf) for inf in Zin.influence])"
        # Other influence matrices remain the same
        # Only need to update the owned generator influence matrix
        if !isnothing(Zin.owned_generators) && Zin.owned_generators == attempt_find_index_position(Zin.generator_ids, ZoutRef.generator_ids[ZoutRef.owned_generators])
            influence_new[ZoutRef.owned_generators][:, 1:column_pos-1] .= Zin.influence[Zin.owned_generators]
        end
        # @debug "Size of owned influence matrix after copy: $(size(influence_new[ZoutRef.owned_generators]))"
        influence_new[ZoutRef.owned_generators][:,column_pos:end] .= 0.0
        bounds_range = upper .- lower
        @inbounds for (idx, g) in enumerate(Zin.Gs)
            influence_new[ZoutRef.owned_generators][:,column_pos:end] .+= Zin.influence[idx] * abs.(g ./ bounds_range)'
        end
    else
        influence_new = Zin.influence
    end

    updateGeneratorsMul!(ZoutRef.Gs, indices, Zin.Gs, λ, :)

    # fill number of new_gens columns with all 0, 
    # then add the error term to the diagonal of these columns
    ZoutRef.Gs[ZoutRef.owned_generators][:,(end-new_gens+1):end] .= 0.0
    generator_offset = size(ZoutRef.Gs[ZoutRef.owned_generators],2) - new_gens
    A = ZoutRef.Gs[ZoutRef.owned_generators]
    @inbounds for (i, row) in enumerate(1:new_gens)
        A[row, (generator_offset + i)] = abs(γ[row])
    end
end

function propagate_layer!(ZoutRefVec :: Vector{Zonotope}, L :: ONNXSigmoid{S}, inputs :: Vector{Zonotope}; lower=nothing, upper=nothing) where {S}
    @assert length(inputs) == 1 "Sigmoid layer should have exactly one input"
    @assert length(ZoutRefVec) == 1 "Sigmoid layer should have exactly one output"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]
    return propagate_layer!(ZoutRef, L, Zin; lower=lower, upper=upper)
end

function σ(x)
    positive = (x .> 0.0)
    negative = .!positive
    result = zeros(length(x))
    result[positive] .= (1 ./ (1 .+ exp.(.- x[positive])))
    result[negative] .= (exp.(x[negative]) ./ (exp.(x[negative]) .+ 1))
    return result
end

function σ´(x)
    return σ(x) .* (1 .- σ(x))
end

function solve_σ´(λ, use_upper)
    if any(λ .== 0)
        throw("λ is zero")
    end
    if use_upper
        return log.((1 .- 2 .* λ  .+ sqrt.(1 .- 4 .* λ)) ./ (2 .* λ))
    else
        return log.((1 .- 2 .* λ .- sqrt.(1 .- 4 .* λ)) ./ (2 .* λ))
    end
end

function fsecant_slope(lower, upper)
    secant_slope = clamp.((σ(upper) .- σ(lower)) ./ (upper .- lower), 0, 0.25)
    secant_slope[secant_slope .< CUTOFF_SIGMOID_SLOPE] .= 0
    return secant_slope
end

function iterate_tagent_point(start, fix_point, use_upper)
    tangent_point = start
    solve_derivative = trues(length(start))
    slope = zeros(length(start))
    for i in 1:LOOP_ITERATIONS_SINGLE_SIGMOID
        slope[solve_derivative] .= fsecant_slope(tangent_point[solve_derivative], fix_point[solve_derivative])
        solve_derivative .= (slope .>= CUTOFF_SIGMOID_SLOPE)
        tangent_point[solve_derivative] .= solve_σ´(slope[solve_derivative], use_upper)
        tangent_point[.!solve_derivative] .= start[.!solve_derivative]
    end
    return tangent_point
end

function propagate_layer!(ZoutRef :: Zonotope, _L :: ONNXSigmoid{S}, Zin :: Zonotope; lower=nothing, upper=nothing) where {S}
    if isnothing(lower) || isnothing(upper)
        bounds = zono_bounds(Zin)
        lower = @view bounds[:,1]
        upper = @view bounds[:,2]
    end
    
    #println("input_lower = $(lower)")
    #println("input_upper = $(upper)")

    dim = length(lower)
    new_gens = dim
    only_center = (lower .== upper) # handling of this case can be optimised

    #setup data structures
    iterative_slope_lower = zeros(dim)
    iterative_slope_upper = zeros(dim)
    iterative_slope = zeros(dim)
    tangent_points = zeros(2, dim) #[lower tangent points ,upper tangent points]

    #calc secant slope for all 
    secant_slope = fsecant_slope(lower, upper)

    #check secant slope vailidity
    upper_derivative = σ´(upper)
    lower_derivative = σ´(lower)
    secant_upper = @simd_bool_expr dim (secant_slope <= upper_derivative)
    secant_lower = @simd_bool_expr dim (secant_slope <= lower_derivative)

    #calc tangent points and slope iteratively
    mask_iteration = (.!(secant_upper .|| secant_lower))
    tangent_points[1, mask_iteration] = iterate_tagent_point(lower[mask_iteration], upper[mask_iteration], false) 
    tangent_points[2, mask_iteration] = iterate_tagent_point(upper[mask_iteration], lower[mask_iteration], true)
    iterative_slope_lower[mask_iteration] = σ´(tangent_points[1, mask_iteration])
    iterative_slope_upper[mask_iteration] = σ´(tangent_points[2, mask_iteration])

    #choose smaller iterative slope
    mask_lower = @simd_bool_expr dim (iterative_slope_lower <= iterative_slope_upper)
    mask_upper = @simd_bool_expr dim (iterative_slope_lower > iterative_slope_upper)
    iterative_slope[mask_lower] = iterative_slope_lower[mask_lower]
    iterative_slope[mask_upper] = iterative_slope_upper[mask_upper]

    #set tangent points where opposite slope is used using σ'(x) = σ'(-x)
    tangent_points[1, mask_upper] = .-tangent_points[2, mask_upper]
    tangent_points[2, mask_lower] = .-tangent_points[1, mask_lower] 

    #set tangent points where secant slope is used
    secant_slope_zero = (secant_slope .== 0) 
    tangent_points[1, secant_upper .&& .!secant_slope_zero] = solve_σ´(secant_slope[secant_upper .&& .!secant_slope_zero], false)
    tangent_points[1, secant_upper .&& secant_slope_zero] = lower[secant_upper .&& secant_slope_zero]
    tangent_points[1, secant_lower] = upper[secant_lower]
    tangent_points[2, secant_upper] = upper[secant_upper]
    tangent_points[2, secant_lower .&& .!secant_slope_zero] = solve_σ´(secant_slope[secant_lower .&& .!secant_slope_zero], true)
    tangent_points[2, secant_lower .&& secant_slope_zero] = upper[secant_lower .&& secant_slope_zero]

    #calc final slope for all
    λ = ifelse.(secant_upper .|| secant_lower, secant_slope, iterative_slope)

    #calc and apply center offset
    lower_tangent_points = @view tangent_points[1, :]
    upper_tangent_points = @view tangent_points[2, :]
    σ_upper = σ(upper_tangent_points)
    σ_lower = σ(lower_tangent_points)
    
    ν = 0.5 .* (.-λ .* upper_tangent_points .+ σ_upper .+ 1e-7 .- λ .* lower_tangent_points .+ σ_lower .- 1e-7)
    ZoutRef.c .= λ .* Zin.c .+ ν
    ZoutRef.c[only_center] .= σ(Zin.c[only_center])

    #calc new generator
    μ = 0.5 .* (.-λ .* upper_tangent_points .+ σ_upper .+ 1e-7 .+ λ .* lower_tangent_points .- σ_lower .+ 1e-7)
    μ[only_center] .= 0 # one additional column per only center affine form 
    

    indices = intersect_indices(ZoutRef.generator_ids, Zin.generator_ids)
    if VeryDiff.NEW_HEURISTIC[]
        influence_new = ZoutRef.influence
        column_pos = size(influence_new[ZoutRef.owned_generators],2) - new_gens + 1
        if !isnothing(Zin.owned_generators) && Zin.owned_generators == attempt_find_index_position(Zin.generator_ids, ZoutRef.generator_ids[ZoutRef.owned_generators])
            influence_new[ZoutRef.owned_generators][:, 1:column_pos-1] .= Zin.influence[Zin.owned_generators]
        end
        influence_new[ZoutRef.owned_generators][:,column_pos:end] .= 0.0
        bounds_range = upper .- lower
        @inbounds for (idx, g) in enumerate(Zin.Gs)
            influence_new[ZoutRef.owned_generators][:,column_pos:end] .+= Zin.influence[idx] * abs.((g) ./ bounds_range)'
        end
    else
        influence_new = Zin.influence
    end

    num_new_gens = dim

    λ[only_center] .= 1
    updateGeneratorsMul!(ZoutRef.Gs, indices, Zin.Gs, λ, :) #apply slope
    ZoutRef.Gs[ZoutRef.owned_generators][:,(end-num_new_gens+1):end] .= 0.0
    generator_offset = size(ZoutRef.Gs[ZoutRef.owned_generators],2) - num_new_gens
    A = ZoutRef.Gs[ZoutRef.owned_generators]
    @inbounds for row in axes(A, 1)
        A[row, (generator_offset + row)] = abs(μ[row]) #add new generators
    end

    #output_bounds = zono_bounds(ZoutRef)
    #output_lower = @view output_bounds[:,1]
    #output_upper = @view output_bounds[:,2]
    #println("output_lower = $(output_lower)")
    #println("output_upper = $(output_upper)")

end