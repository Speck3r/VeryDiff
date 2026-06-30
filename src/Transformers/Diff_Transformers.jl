function propagate_layer!(
    ZoutRefVec :: Vector{CachedZonotope},
    Ls :: DiffLayer{
        ONNXLinear{S1},
        ONNXLinear{S2},
        ONNXLinear{S3}},
    inputs :: Vector{DiffZonotope};
    bounds_cache :: Union{Nothing,BoundsCache}=nothing) where {S1, S2, S3}
    @assert length(inputs) == 1 "Dense layer should have exactly one input zonotope"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output zonotope"
    ZoutRef = ZoutRefVec[1]
    # @debug "Propagating DiffDense Layer"
    Zin = inputs[1]
    # Compute differential zonotope dimensions
    # TODO(steuber): Is there a more elegant way?
    # At the very least we could probably extract this into a function
    i_out = 1
    ∂g_dims = Int64[]
    resize!(∂g_dims, length(ZoutRef.zonotope_proto.∂Z.Gs))
    for i_out in 1:length(∂g_dims)
        res = attempt_find_index_position(Zin.∂Z.generator_ids, ZoutRef.zonotope_proto.∂Z.generator_ids[i_out])
        if res > 0
            ∂g_dims[i_out] = size(Zin.∂Z.Gs[res],2)
        else
            res = find_index_position(Zin.Z₂.generator_ids, ZoutRef.zonotope_proto.∂Z.generator_ids[i_out])
            ∂g_dims[i_out] = size(Zin.Z₂.Gs[res],2)
        end
    end
    Zout = get_zonotope!(ZoutRef, size.(Zin.Z₁.Gs,2), size.(Zin.Z₂.Gs,2), ∂g_dims)
    L1 = get_layer1(Ls)
    ∂L = get_diff_layer(Ls)
    L2 = get_layer2(Ls)
    L1_W = L1.dense.weight
    ∂L_W = ∂L.dense.weight
    ∂L_b = ∂L.dense.bias
    if VeryDiff.USE_DIFFZONO[]
        # @debug "IDs of Output Zonotope Generators: $(Zout.∂Z.generator_ids)"
        ∂indices = intersect_indices(Zout.∂Z.generator_ids, Zin.∂Z.generator_ids)
        for (i, g) in zip(∂indices, Zin.∂Z.Gs)
            mul!(Zout.∂Z.Gs[i], L1_W, g)
        end
        indices₂ = intersect_indices(Zout.∂Z.generator_ids, Zin.Z₂.generator_ids)
        for (i, g) in zip(indices₂, Zin.Z₂.Gs)
            mul!(Zout.∂Z.Gs[i], ∂L_W, g, 1.0, 1.0)
        end
        # @assert length(intersect_indices(Zout.∂Z.generator_ids, union(Zin.∂Z.generator_ids, Zin.Z₂.generator_ids))) == length(Zout.∂Z.generator_ids) "Not all generators in ∂Z were processed during Dense propagation!"
        mul!(Zout.∂Z.c, L1_W, Zin.∂Z.c)
        mul!(Zout.∂Z.c, ∂L_W, Zin.Z₂.c, 1.0, 1.0)
        Zout.∂Z.c .+= ∂L_b
    end
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂)
end

function propagate_layer!(
    ZoutRefVec :: Vector{CachedZonotope},
    Ls :: DiffLayer{
        ONNXAddConst{S1},
        ONNXAddConst{S2},
        ONNXAddConst{S3}},
    inputs :: Vector{DiffZonotope};
    bounds_cache :: Union{Nothing,BoundsCache}=nothing) where {S1, S2, S3}
    @assert length(inputs) == 1 "Dense layer should have exactly one input zonotope"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output zonotope"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]
    i_out = 1
    ∂g_dims = Int64[]
    resize!(∂g_dims, length(ZoutRef.zonotope_proto.∂Z.Gs))
    for i_out in 1:length(∂g_dims)
        res = attempt_find_index_position(Zin.∂Z.generator_ids, ZoutRef.zonotope_proto.∂Z.generator_ids[i_out])
        if res > 0
            ∂g_dims[i_out] = size(Zin.∂Z.Gs[res],2)
        else
            res = find_index_position(Zin.Z₂.generator_ids, ZoutRef.zonotope_proto.∂Z.generator_ids[i_out])
            ∂g_dims[i_out] = size(Zin.Z₂.Gs[res],2)
        end
    end
    Zout = get_zonotope!(ZoutRef, size.(Zin.Z₁.Gs,2), size.(Zin.Z₂.Gs,2), ∂g_dims)
    L1 = get_layer1(Ls)
    ∂L = get_diff_layer(Ls)
    ∂L_b = ∂L.c
    L2 = get_layer2(Ls)
    if VeryDiff.USE_DIFFZONO[]
        # @debug "IDs of Output Zonotope Generators: $(Zout.∂Z.generator_ids)"
        ∂indices = intersect_indices(Zout.∂Z.generator_ids, Zin.∂Z.generator_ids)
        for (i, g) in zip(∂indices, Zin.∂Z.Gs)
            Zout.∂Z.Gs[i] .= g
        end
        Zout.∂Z.c .= Zin.∂Z.c .+ ∂L_b
    end
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂)
end

function propagate_layer!(
    ZoutRefVec :: Vector{CachedZonotope},
    Ls :: DiffLayer{
        ONNXLinear{S1},
        ZeroDense{S2},
        ONNXLinear{S3}},
    inputs :: Vector{DiffZonotope};
    bounds_cache :: Union{Nothing,BoundsCache}=nothing) where {S1, S2, S3}
    @assert length(inputs) == 1 "Dense layer should have exactly one input zonotope"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output zonotope"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]
    Zout = get_zonotope!(ZoutRef, size.(Zin.Z₁.Gs,2), size.(Zin.Z₂.Gs,2), convert(Vector{Int64},size.(Zin.∂Z.Gs,2)))
    L1 = get_layer1(Ls)
    L2 = get_layer2(Ls)
    L1_W = L1.dense.weight
    L2_W = L2.dense.weight
    L1_b = L1.dense.bias
    L2_b = L2.dense.bias
    if VeryDiff.USE_DIFFZONO[]
        ∂indices = intersect_indices(Zout.∂Z.generator_ids, Zin.∂Z.generator_ids)
        # @assert length(union(Zout.∂Z.generator_ids,Zin.∂Z.generator_ids)) == length(Zout.∂Z.generator_ids) "Not all generators in ∂Z were processed during Dense propagation. Output IDs: $(Zout.∂Z.generator_ids), Processed IDs: $(Zin.∂Z.generator_ids)"
        for (i, g) in zip(∂indices, Zin.∂Z.Gs)
            mul!(Zout.∂Z.Gs[i], L1_W, g)
        end
        mul!(Zout.∂Z.c, L1_W, Zin.∂Z.c)
    end
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂)
end

function α(lower, upper)
    return (.-lower ./ (upper .- lower))
end

function ∂λ(∂lower, ∂upper, lower₁, upper₁)
    return ifelse.(∂lower .>= 0, 0.0, ifelse.(∂upper .<= 0, 1.0, min.(∂upper, upper₁) ./ (min.(∂upper, upper₁) .- max.(∂lower, lower₁))))
end

function μ(lower, upper)
    return (0.5 .* α(lower, upper) .* upper)
end

function ∂μ(∂lower, ∂upper, lower₁, upper₁) 
    cur_∂λ = ∂λ(∂lower, ∂upper, lower₁, upper₁)
    return 0.5.*(max.(cur_∂λ .* ∂lower .- ∂lower,cur_∂λ .* ∂upper) .+ ifelse.(∂lower .>= 0,min.(∂upper,upper₁),.-cur_∂λ .* max.(∂lower,lower₁)))
end

function ∂ν(∂lower, ∂upper, lower₁, upper₁)
    cur_∂λ = ∂λ(∂lower, ∂upper, lower₁, upper₁)
    return 0.5.*(.-max.(cur_∂λ .* ∂lower .- ∂lower,cur_∂λ .* ∂upper) .+ ifelse.(∂lower .>= 0,min.(∂upper,upper₁),.-cur_∂λ .* max.(∂lower,lower₁)))
end

function ∂a(any_any, ∂lower, ∂upper, lower₁, upper₁) 
    return ifelse(any_any, ∂λ(∂lower, ∂upper, lower₁, upper₁), 1.0)
end

function propagate_layer!(
    ZoutRefVec :: Vector{CachedZonotope},
    Ls :: DiffLayer{
        ONNXRelu{S1},
        ONNXRelu{S2},
        ONNXRelu{S3}},
    inputs :: Vector{DiffZonotope};
    bounds_cache :: Union{Nothing,BoundsCache}=nothing) where {S1, S2, S3}
    @assert length(inputs) == 1 "ReLU layer should have exactly one input zonotope"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output zonotope"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]

    @assert !isnothing(bounds_cache)

    # Compute Bounds
    bounds₁ = zono_bounds(Zin.Z₁)
    bounds₂ = zono_bounds(Zin.Z₂)
    ∂bounds = zono_bounds(Zin.∂Z)

    if !bounds_cache.initialized
        bounds_cache.lower₁ = copy(bounds₁[:,1])
        bounds_cache.upper₁ = copy(bounds₁[:,2])
        bounds_cache.lower₂ = copy(bounds₂[:,1])
        bounds_cache.upper₂ = copy(bounds₂[:,2])
        bounds_cache.∂lower = copy(∂bounds[:,1])
        bounds_cache.∂upper = copy(∂bounds[:,2])
        bounds_cache.initialized = true
    else
        bounds_cache.lower₁ .= max.(bounds₁[:,1], bounds_cache.lower₁)
        bounds_cache.upper₁ .= min.(bounds₁[:,2], bounds_cache.upper₁)
        bounds_cache.lower₂ .= max.(bounds₂[:,1], bounds_cache.lower₂)
        bounds_cache.upper₂ .= min.(bounds₂[:,2], bounds_cache.upper₂)
        bounds_cache.∂lower .= max.(∂bounds[:,1], bounds_cache.∂lower)
        bounds_cache.∂upper .= min.(∂bounds[:,2], bounds_cache.∂upper)
    end
    lower₁ = bounds_cache.lower₁
    upper₁ = bounds_cache.upper₁
    lower₂ = bounds_cache.lower₂
    upper₂ = bounds_cache.upper₂
    ∂lower = bounds_cache.∂lower
    ∂upper = bounds_cache.∂upper
    #@info "Bounds Cache: Z₁=[$(lower₁), $(upper₁)], Z₂=[$(lower₂), $(upper₂)], ∂Z=[$(∂lower), $(∂upper)]"

    (
        zero_diff,
        neg_neg,
        neg_pos,
        pos_neg,
        pos_pos,
        any_neg,
        neg_any,
        any_pos,
        pos_any,
        any_any
    ) = get_selectors(bounds₁, bounds₂, ∂bounds)
    # Do NOT use counts created above for new_gen₁ / new_gen₂,
    # because these omit dimensions where difference is still zero
    new_gen₁ = count(lower₁ .< 0.0 .&& upper₁ .> 0.0)
    new_gen₂ = count(lower₂ .< 0.0 .&& upper₂ .> 0.0)
    ∂new_gen = count(any_pos) + count(pos_any) + count(any_any)
    # @debug "Instable Neurons: Network 1: $new_gen₁, Network 2: $new_gen₂, Differential: $∂new_gen"
    Zout_proto = ZoutRef.zonotope_proto # Need this to be able to access the generator ids
    gen_sizes₁ = zeros(Int64,length(Zout_proto.Z₁.generator_ids))
    gen_sizes₂ = zeros(Int64,length(Zout_proto.Z₂.generator_ids))
    ∂gen_sizes = zeros(Int64,length(Zout_proto.∂Z.generator_ids))

    pre_indices_Z₁ = intersect_indices(Zout_proto.Z₁.generator_ids, Zin.Z₁.generator_ids)
    pre_indices_Z₂ = intersect_indices(Zout_proto.Z₂.generator_ids, Zin.Z₂.generator_ids)
    pre_indices₁ = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.Z₁.generator_ids)
    pre_indices₂ = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.Z₂.generator_ids)
    ∂pre_indices = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.∂Z.generator_ids)

    for (i, idx) in enumerate(pre_indices_Z₁)
        gen_sizes₁[idx] = size(Zin.Z₁.Gs[i],2)
    end
    gen_sizes₁[Zout_proto.Z₁.owned_generators] += new_gen₁
    for (i, idx) in enumerate(pre_indices_Z₂)
        gen_sizes₂[idx] = size(Zin.Z₂.Gs[i],2)
    end
    gen_sizes₂[Zout_proto.Z₂.owned_generators] += new_gen₂
    # This mayoverwrite sizes, but columns should be consistent
    # TODO(steuber): Can we make this cleaner?
    for (i, idx) in enumerate(∂pre_indices)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.∂Z.Gs[i],2)) (from ∂Z)"
        ∂gen_sizes[idx] = size(Zin.∂Z.Gs[i],2)
    end
    for (i, idx) in enumerate(pre_indices₁)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.Z₁.Gs[i],2)) (from Z₁)"
        ∂gen_sizes[idx] = size(Zin.Z₁.Gs[i],2)
    end
    for (i, idx) in enumerate(pre_indices₂)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.Z₂.Gs[i],2)) (from Z₂)"
        ∂gen_sizes[idx] = size(Zin.Z₂.Gs[i],2)
    end
    # @info "Generator sizes before new gens: Z₁=$(gen_sizes₁), Z₂=$(gen_sizes₂), ∂Z=$(∂gen_sizes)"
    ∂old_gen = ∂gen_sizes[Zout_proto.∂Z.owned_generators]
    ∂gen_sizes[Zout_proto.∂Z.owned_generators] += ∂new_gen
    # Find idx of generators owned by Z₁ and Z₂ in ∂Z
    idx1 = find_index_position(Zout_proto.∂Z.generator_ids, Zout_proto.Z₁.generator_ids[Zout_proto.Z₁.owned_generators])
    idx2 = find_index_position(Zout_proto.∂Z.generator_ids, Zout_proto.Z₂.generator_ids[Zout_proto.Z₂.owned_generators])
    ∂gen_sizes[idx1] += new_gen₁
    ∂gen_sizes[idx2] += new_gen₂
    Zout_proto = nothing # Avoid missuse
    # @info "ReLU DiffZonotope Generators: Z₁=$(gen_sizes₁), Z₂=$(gen_sizes₂), ∂Z=$(∂gen_sizes)"
    Zout = get_zonotope!(ZoutRef, gen_sizes₁, gen_sizes₂, ∂gen_sizes)
    post_indices₁ = intersect_indices(Zout.∂Z.generator_ids, Zout.Z₁.generator_ids)
    post_indices₂ = intersect_indices(Zout.∂Z.generator_ids, Zout.Z₂.generator_ids)

    L1 = get_layer1(Ls)
    L2 = get_layer2(Ls)
    # Compute Zonotopes for individual networks
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁;lower=lower₁, upper=upper₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂;lower=lower₂, upper=upper₂)

    if VeryDiff.USE_DIFFZONO[]
        dim = length(any_neg)
        â₁_pos = @simd_bool_expr dim (any_neg | pos_neg)
        a₁_pos = any_pos
        â₂_pos = @simd_bool_expr dim (neg_any | neg_pos)
        a₂_pos = pos_any
        ∂a_pos_∂λ = any_any
        # This one *must* be addition
        ∂a_pos_1 = @simd_bool_expr dim (any_pos | pos_any)
        ∂a_pos_1_assign = pos_pos
        
        # Reset to zero
        Zout.∂Z.c .= 0.0
        selector = @simd_bool_expr dim (neg_neg | zero_diff)
        for g in Zout.∂Z.Gs
            g[selector, :] .= 0.0
        end
        
        # Assign Zin.Z₁ with a₁
        cur_α₁ = .-α((@view lower₁[a₁_pos]), (@view upper₁[a₁_pos]))
        updateGeneratorsMul!(Zout.∂Z.Gs, pre_indices₁, Zin.Z₁.Gs, cur_α₁, a₁_pos)
        Zout.∂Z.c[a₁_pos] .= cur_α₁ .* (@view Zin.Z₁.c[a₁_pos])

        # Assign Zout.Z₁ with â₁ = 1
        updateGenerators!(Zout.∂Z.Gs, post_indices₁, Zout.Z₁.Gs, â₁_pos)
        Zout.∂Z.c[â₁_pos] .= (@view Zout.Z₁.c[â₁_pos])

        # Assign Zin.Z₂ with a₂
        cur_α₂ = α((@view lower₂[a₂_pos]), (@view upper₂[a₂_pos]))
        updateGeneratorsMul!(Zout.∂Z.Gs, pre_indices₂, Zin.Z₂.Gs, cur_α₂, a₂_pos)
        Zout.∂Z.c[a₂_pos] .= cur_α₂ .* (@view Zin.Z₂.c[a₂_pos])

        # Assign Zout.Z₂ with â₂ = -1
        updateGeneratorsMul!(Zout.∂Z.Gs, post_indices₂, Zout.Z₂.Gs, -1.0, â₂_pos)
        Zout.∂Z.c[â₂_pos] .= .-(@view Zout.Z₂.c[â₂_pos])

        # Add Zin.∂Z with 1.0
        updateGeneratorsAdd!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, ∂a_pos_1)
        Zout.∂Z.c[∂a_pos_1] .+= (@view Zin.∂Z.c[∂a_pos_1])

        # Assign Zin.∂Z with 1.0
        updateGenerators!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, ∂a_pos_1_assign)
        Zout.∂Z.c[∂a_pos_1_assign] .= (@view Zin.∂Z.c[∂a_pos_1_assign])

        # Add Zin.∂Z with ∂λ
        cur_∂λ = ∂λ((@view ∂lower[any_any]), (@view ∂upper[any_any]),(@view lower₁[any_any]),(@view upper₁[any_any]))
        # TODO(steuber): Add requires copy vs. assign does not!
        updateGeneratorsMul!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, cur_∂λ, ∂a_pos_∂λ)
        Zout.∂Z.c[∂a_pos_∂λ] .= cur_∂λ .* (@view Zin.∂Z.c[∂a_pos_∂λ])

        # Add new generators from c
        c_pos = findall(@simd_bool_expr dim (any_any | any_pos | pos_any))
        A = Zout.∂Z.Gs[Zout.∂Z.owned_generators]
        @inbounds for i in 1:length(c_pos)
            row = c_pos[i]
            col = ∂old_gen + i
            if any_any[row]
                A[row, col] = ∂μ(∂lower[row], ∂upper[row], lower₁[row], upper₁[row])
            elseif any_pos[row]
                A[row, col] = μ(lower₁[row], upper₁[row])
            else # pos_any[row]
                A[row, col] = μ(lower₂[row], upper₂[row])
            end
        end

        # Add bias
        Zout.∂Z.c .+= ifelse.(
            any_any, ∂ν(∂lower, ∂upper, lower₁, upper₁),
                ifelse.(any_pos, μ(lower₁,upper₁),
                    ifelse.(pos_any, .-μ(lower₂,upper₂), 0.0)))
    end
end

function f_mu(lambda, row)
    return (0.5 .* lambda .* row)
end

function f_lambda_any_pos(lower₁, upper₁, alpha)
    return ((lower₁ .- (alpha .* lower₁)) ./ (upper₁ .- lower₁))
end

function f_lambda_pos_any(lower₂, upper₂, alpha)
    return ((.-lower₂ .+ (alpha .* lower₂)) ./ (upper₂ .- lower₂))
end

function f_lambda_any_neg(lower₁, upper₁, alpha)
    return ((upper₁ .- (alpha .* upper₁)) ./ (upper₁ .- lower₁))
end

function f_lambda_neg_any(lower₂, upper₂, alpha)
    return ((.-upper₂ .+ (alpha .* upper₂)) ./ (upper₂ .- lower₂))
end

function slope_any_any(minimal_upper, maximal_lower, alpha)
    return (minimal_upper .- (alpha .* maximal_lower)) ./ (minimal_upper .- maximal_lower)
end

function f_lambda_any_any(∂lower, ∂upper, lower₁, upper₁, alpha)
    minimal_upper = min.(∂upper, upper₁)
    maximal_lower = max.(∂lower, lower₁)
    return ifelse.((minimal_upper .- maximal_lower) .== 0, ifelse.(∂lower .>= 0, alpha, 1.0), ifelse.(∂lower .>= 0, alpha, ifelse.(∂upper .<= 0, 1.0, slope_any_any(minimal_upper, maximal_lower, alpha))))
end

function f_mu_any_any(∂lower, ∂upper, lower₁, upper₁, alpha)
    alpha_check = alpha > 1 ? 2 : 1
    minimal_upper = min.(∂upper, upper₁)
    maximal_lower = max.(∂lower, lower₁)
    lambda_any_any = f_lambda_any_any(∂lower, ∂upper, lower₁, upper₁, alpha)
    mu_delta_pos = (1-alpha) .* minimal_upper
    mu_delta_neg =  (alpha .- lambda_any_any) .* maximal_lower 
    mu_otherwise = .-((-1)^alpha_check) .* max.(abs.((lambda_any_any .- 1) .*  ∂lower),abs.((lambda_any_any .- alpha) .*  ∂upper)) .+ mu_delta_neg
    return 0.5 .* ifelse.(∂lower .>= 0, mu_delta_pos, ifelse.(∂upper .<= 0, mu_delta_neg, mu_otherwise))
end

function f_nu_any_any(∂lower, ∂upper, lower₁, upper₁, alpha)
    alpha_check = alpha > 1 ? 2 : 1
    minimal_upper = min.(∂upper, upper₁)
    maximal_lower = max.(∂lower, lower₁)
    lambda_any_any = f_lambda_any_any(∂lower, ∂upper, lower₁, upper₁, alpha)
    nu_delta_pos = (1-alpha) .* minimal_upper
    nu_delta_neg =  (alpha .- lambda_any_any) .* maximal_lower 
    nu_otherwise = ((-1)^alpha_check) .* max.(abs.((lambda_any_any .- 1) .*  ∂lower),abs.((lambda_any_any .- alpha) .*  ∂upper)) .+ nu_delta_neg
    return 0.5 .* ifelse.(∂lower .>= 0, nu_delta_pos, ifelse.(∂upper .<= 0, nu_delta_neg, nu_otherwise))
end

function propagate_layer!(
    ZoutRefVec :: Vector{CachedZonotope},
    Ls :: DiffLayer{
        ONNXLeakyRelu{S1, F1},
        ONNXLeakyRelu{S2, F2},
        ONNXLeakyRelu{S3, F3}},
    inputs :: Vector{DiffZonotope};
    bounds_cache :: Union{Nothing,BoundsCache}=nothing) where {S1, S2, S3, F1, F2, F3}
    @assert length(inputs) == 1 "LeakyReLU layer should have exactly one input zonotope"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output zonotope"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]

    @assert !isnothing(bounds_cache)

    # Compute Bounds
    bounds₁ = zono_bounds(Zin.Z₁)
    bounds₂ = zono_bounds(Zin.Z₂)
    ∂bounds = zono_bounds(Zin.∂Z)

    if !bounds_cache.initialized
        bounds_cache.lower₁ = copy(bounds₁[:,1])
        bounds_cache.upper₁ = copy(bounds₁[:,2])
        bounds_cache.lower₂ = copy(bounds₂[:,1])
        bounds_cache.upper₂ = copy(bounds₂[:,2])
        bounds_cache.∂lower = copy(∂bounds[:,1])
        bounds_cache.∂upper = copy(∂bounds[:,2])
        bounds_cache.initialized = true
    else
        bounds_cache.lower₁ .= max.(bounds₁[:,1], bounds_cache.lower₁)
        bounds_cache.upper₁ .= min.(bounds₁[:,2], bounds_cache.upper₁)
        bounds_cache.lower₂ .= max.(bounds₂[:,1], bounds_cache.lower₂)
        bounds_cache.upper₂ .= min.(bounds₂[:,2], bounds_cache.upper₂)
        bounds_cache.∂lower .= max.(∂bounds[:,1], bounds_cache.∂lower)
        bounds_cache.∂upper .= min.(∂bounds[:,2], bounds_cache.∂upper)
    end
    lower₁ = bounds_cache.lower₁
    upper₁ = bounds_cache.upper₁
    lower₂ = bounds_cache.lower₂
    upper₂ = bounds_cache.upper₂
    ∂lower = bounds_cache.∂lower
    ∂upper = bounds_cache.∂upper
    #@info "Bounds Cache: Z₁=[$(lower₁), $(upper₁)], Z₂=[$(lower₂), $(upper₂)], ∂Z=[$(∂lower), $(∂upper)]"

    (
        zero_diff,
        neg_neg,
        neg_pos,
        pos_neg,
        pos_pos,
        any_neg,
        neg_any,
        any_pos,
        pos_any,
        any_any
    ) = get_selectors(bounds₁, bounds₂, ∂bounds)
    # Do NOT use counts created above for new_gen₁ / new_gen₂,
    # because these omit dimensions where difference is still zero
    new_gen₁ = count(lower₁ .< 0.0 .&& upper₁ .> 0.0)
    new_gen₂ = count(lower₂ .< 0.0 .&& upper₂ .> 0.0)
    ∂new_gen = count(any_neg) + count(neg_any) + count(any_pos) + count(pos_any) + count(any_any)
    # @debug "Instable Neurons: Network 1: $new_gen₁, Network 2: $new_gen₂, Differential: $∂new_gen"
    Zout_proto = ZoutRef.zonotope_proto # Need this to be able to access the generator ids
    gen_sizes₁ = zeros(Int64,length(Zout_proto.Z₁.generator_ids))
    gen_sizes₂ = zeros(Int64,length(Zout_proto.Z₂.generator_ids))
    ∂gen_sizes = zeros(Int64,length(Zout_proto.∂Z.generator_ids))

    pre_indices_Z₁ = intersect_indices(Zout_proto.Z₁.generator_ids, Zin.Z₁.generator_ids)
    pre_indices_Z₂ = intersect_indices(Zout_proto.Z₂.generator_ids, Zin.Z₂.generator_ids)
    pre_indices₁ = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.Z₁.generator_ids)
    pre_indices₂ = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.Z₂.generator_ids)
    ∂pre_indices = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.∂Z.generator_ids)

    for (i, idx) in enumerate(pre_indices_Z₁)
        gen_sizes₁[idx] = size(Zin.Z₁.Gs[i],2)
    end
    gen_sizes₁[Zout_proto.Z₁.owned_generators] += new_gen₁
    for (i, idx) in enumerate(pre_indices_Z₂)
        gen_sizes₂[idx] = size(Zin.Z₂.Gs[i],2)
    end
    gen_sizes₂[Zout_proto.Z₂.owned_generators] += new_gen₂
    # This mayoverwrite sizes, but columns should be consistent
    # TODO(steuber): Can we make this cleaner?
    for (i, idx) in enumerate(∂pre_indices)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.∂Z.Gs[i],2)) (from ∂Z)"
        ∂gen_sizes[idx] = size(Zin.∂Z.Gs[i],2)
    end
    for (i, idx) in enumerate(pre_indices₁)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.Z₁.Gs[i],2)) (from Z₁)"
        ∂gen_sizes[idx] = size(Zin.Z₁.Gs[i],2)
    end
    for (i, idx) in enumerate(pre_indices₂)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.Z₂.Gs[i],2)) (from Z₂)"
        ∂gen_sizes[idx] = size(Zin.Z₂.Gs[i],2)
    end
    # @info "Generator sizes before new gens: Z₁=$(gen_sizes₁), Z₂=$(gen_sizes₂), ∂Z=$(∂gen_sizes)"
    ∂old_gen = ∂gen_sizes[Zout_proto.∂Z.owned_generators]
    ∂gen_sizes[Zout_proto.∂Z.owned_generators] += ∂new_gen
    # Find idx of generators owned by Z₁ and Z₂ in ∂Z
    idx1 = find_index_position(Zout_proto.∂Z.generator_ids, Zout_proto.Z₁.generator_ids[Zout_proto.Z₁.owned_generators])
    idx2 = find_index_position(Zout_proto.∂Z.generator_ids, Zout_proto.Z₂.generator_ids[Zout_proto.Z₂.owned_generators])
    ∂gen_sizes[idx1] += new_gen₁
    ∂gen_sizes[idx2] += new_gen₂
    Zout_proto = nothing # Avoid missuse
    # @info "LeakyReLU DiffZonotope Generators: Z₁=$(gen_sizes₁), Z₂=$(gen_sizes₂), ∂Z=$(∂gen_sizes)"
    Zout = get_zonotope!(ZoutRef, gen_sizes₁, gen_sizes₂, ∂gen_sizes)
    post_indices₁ = intersect_indices(Zout.∂Z.generator_ids, Zout.Z₁.generator_ids)
    post_indices₂ = intersect_indices(Zout.∂Z.generator_ids, Zout.Z₂.generator_ids)

    L1 = get_layer1(Ls)
    L2 = get_layer2(Ls)
    # Compute Zonotopes for individual networks
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁;lower=lower₁, upper=upper₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂;lower=lower₂, upper=upper₂)

    if VeryDiff.USE_DIFFZONO[]

        alpha = convert(Float64,L1.alpha)
        
        # Reset to zero
        Zout.∂Z.c .= 0.0
        for g in Zout.∂Z.Gs
            g[zero_diff, :] .= 0.0
        end

        # pos_pos case deltaZout := deltaZin
        updateGenerators!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, pos_pos)
        Zout.∂Z.c[pos_pos] .= (@view Zin.∂Z.c[pos_pos])

        # pos_neg case deltaZout := deltaZin + (1-alpha)Z2in
        updateGenerators!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, pos_neg)
        updateGeneratorsAddMul!(Zout.∂Z.Gs, pre_indices₂, Zin.Z₂.Gs, (1-alpha), pos_neg)
        Zout.∂Z.c[pos_neg] .= (@view Zin.∂Z.c[pos_neg]) + ((1-alpha) .* (@view Zin.Z₂.c[pos_neg]))

        # neg_pos case deltaZout := deltaZin + (alpha-1)Z1in 
        updateGenerators!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, neg_pos)
        updateGeneratorsAddMul!(Zout.∂Z.Gs, pre_indices₁, Zin.Z₁.Gs, (alpha-1), neg_pos)
        Zout.∂Z.c[neg_pos] .= (@view Zin.∂Z.c[neg_pos]) + ((alpha-1) .* (@view Zin.Z₁.c[neg_pos]))

        # neg_neg case deltaZout := alpha*deltaZin
        updateGeneratorsMul!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, alpha, neg_neg)
        Zout.∂Z.c[neg_neg] .= alpha .* (@view Zin.∂Z.c[neg_neg])

        #any_pos case
        lambda_any_pos = f_lambda_any_pos((@view lower₁[any_pos]), (@view upper₁[any_pos]), alpha)
        mu_any_pos = f_mu(-lambda_any_pos, (@view upper₁[any_pos]))
        nu_any_pos = mu_any_pos
        updateGenerators!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, any_pos)
        updateGeneratorsAddMul!(Zout.∂Z.Gs, pre_indices₁, Zin.Z₁.Gs, lambda_any_pos, any_pos)
        Zout.∂Z.c[any_pos] .= (@view Zin.∂Z.c[any_pos]) .+ (lambda_any_pos .* (@view Zin.Z₁.c[any_pos])) .+ nu_any_pos

        #pos_any case
        lambda_pos_any = f_lambda_pos_any((@view lower₂[pos_any]), (@view upper₂[pos_any]), alpha)
        mu_pos_any = f_mu(lambda_pos_any, (@view upper₂[pos_any]))
        nu_pos_any = -mu_pos_any
        updateGenerators!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, pos_any)
        updateGeneratorsAddMul!(Zout.∂Z.Gs, pre_indices₂, Zin.Z₂.Gs, lambda_pos_any, pos_any)
        Zout.∂Z.c[pos_any] .= (@view Zin.∂Z.c[pos_any]) .+ (lambda_pos_any .* (@view Zin.Z₂.c[pos_any])) .+ nu_pos_any

        #any_neg case
        lambda_any_neg = f_lambda_any_neg((@view lower₁[any_neg]), (@view upper₁[any_neg]), alpha)
        mu_any_neg = f_mu(-lambda_any_neg, (@view lower₁[any_neg]))
        nu_any_neg = mu_any_neg
        updateGeneratorsMul!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, alpha, any_neg)
        updateGeneratorsAddMul!(Zout.∂Z.Gs, pre_indices₁, Zin.Z₁.Gs, lambda_any_neg, any_neg)
        Zout.∂Z.c[any_neg] .= (alpha .* (@view Zin.∂Z.c[any_neg])) .+ (lambda_any_neg .* (@view Zin.Z₁.c[any_neg])) .+ nu_any_neg

        #neg_any case
        lambda_neg_any = f_lambda_neg_any((@view lower₂[neg_any]), (@view upper₂[neg_any]), alpha)
        mu_neg_any = f_mu(lambda_neg_any, (@view lower₂[neg_any]))
        nu_neg_any = -mu_neg_any
        updateGeneratorsMul!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, alpha, neg_any)
        updateGeneratorsAddMul!(Zout.∂Z.Gs, pre_indices₂, Zin.Z₂.Gs, lambda_neg_any, neg_any)
        Zout.∂Z.c[neg_any] .= (alpha .* (@view Zin.∂Z.c[neg_any])) .+ (lambda_neg_any .* (@view Zin.Z₂.c[neg_any])) .+ nu_neg_any

        #any_any case
        lambda_any_any = f_lambda_any_any((@view ∂lower[any_any]), (@view ∂upper[any_any]), (@view lower₁[any_any]), (@view upper₁[any_any]), alpha)
        nu_any_any = f_nu_any_any((@view ∂lower[any_any]), (@view ∂upper[any_any]), (@view lower₁[any_any]), (@view upper₁[any_any]), alpha)
        updateGeneratorsMul!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, lambda_any_any, any_any)
        Zout.∂Z.c[any_any] .= (lambda_any_any .* (@view Zin.∂Z.c[any_any])) .+ nu_any_any
        

        # Add new generators from c
        dim = length(any_neg)
        c_pos = findall(@simd_bool_expr dim (any_pos | pos_any | any_neg | neg_any | any_any ))
        A = Zout.∂Z.Gs[Zout.∂Z.owned_generators]
        @inbounds for i in 1:length(c_pos)
            row = c_pos[i]
            col = ∂old_gen + i
            if any_pos[row]
                A[row, col] = f_mu((-f_lambda_any_pos(lower₁[row], upper₁[row], alpha)), upper₁[row])
            elseif pos_any[row]
                A[row, col] = f_mu((f_lambda_pos_any(lower₂[row], upper₂[row], alpha)), upper₂[row])
            elseif any_neg[row]
                A[row, col] = f_mu((-f_lambda_any_neg(lower₁[row], upper₁[row], alpha)), lower₁[row])
            elseif neg_any[row]
                A[row, col] = f_mu((f_lambda_neg_any(lower₂[row], upper₂[row], alpha)), lower₂[row])
            elseif any_any[row]
                A[row, col] = f_mu_any_any(∂lower[row], ∂upper[row], lower₁[row], upper₁[row], alpha)
            end
        end
    end
end

function σ_diff(x, y)
    return σ(x) .- σ(x .- y)
end

function extremest_σ_diff(y)
    return 2 .* σ(y ./ 2) .- 1
end

function ∂σ_diff_∂y(x,y)
    return exp.(y .- x) ./ ((exp.(y .- x) .+ 1) .^2)
end

function ∂σ_diff_∂y_extremest(y)
    return exp.(y ./ 2) ./ ((exp.(y ./ 2) .+ 1) .^2)
end

function σ_nondiff(x,y)
    return σ(x) .- σ(y)
end

function ∂σ_nondiff_∂y(x, y)
    return (.-exp.(.-y)) ./ ((1 .+ exp.(.-y)) .^2)
end

function ∂σ_nondiff_∂x(x, y)
    return (exp.(.-x)) ./ ((1 .+ exp.(.-x)) .^2)
end

function solve_∂σ_nondiff_∂y_upper(λ)
    return log.((-2 .* λ .+ sqrt.(4 .* λ .+ 1) .- 1) ./ (2 .* λ))
end

function solve_∂σ_nondiff_∂y_lower(λ)
    return log.((-2 .* λ .- sqrt.(4 .* λ .+ 1) .- 1) ./ (2 .* λ))
end

function solve_∂σ_nondiff_∂x_upper(λ)
    return log.((-2 .* λ .+ sqrt.(1 .- 4 .* λ) .+ 1) ./ (2 .* λ))
end

function solve_∂σ_nondiff_∂x_lower(λ)
    return log.((-2 .* λ .- sqrt.(1 .- 4 .* λ) .+ 1) ./ (2 .* λ))
end

function iterate_nondiff_x_upper!(tangent_points, λ, lower₁, upper₁, lower₂, upper₂)
    iteration = fill(true, length(lower₁))
    for i in 1:10
        λ[iteration] = clamp.(fslope_x_upper((@view tangent_points[iteration]), (@view lower₁[iteration]), (@view upper₁[iteration]), (@view lower₂[iteration]), (@view upper₂[iteration])),0,0.25)
        no_iteration = (abs.(λ) .< CUTOFF_SIGMOID_SLOPE)
        iteration = .!no_iteration
        tangent_points[iteration] = solve_∂σ_nondiff_∂x_upper((@view λ[iteration]))
        λ[no_iteration] .= 0
        tangent_points[no_iteration] = upper₁[no_iteration]
    end
end

function iterate_nondiff_x_lower!(tangent_points, λ, lower₁, upper₁, lower₂, upper₂)
    iteration = fill(true, length(lower₁))
    for i in 1:10
        λ[iteration] = clamp.(fslope_x_lower((@view tangent_points[iteration]), (@view lower₁[iteration]), (@view upper₁[iteration]), (@view lower₂[iteration]), (@view upper₂[iteration])),0,0.25)
        no_iteration = (abs.(λ) .< CUTOFF_SIGMOID_SLOPE)
        iteration = .!no_iteration
        tangent_points[iteration] = solve_∂σ_nondiff_∂x_lower((@view λ[iteration]))
        λ[no_iteration] .= 0
        tangent_points[no_iteration] = lower₁[no_iteration]
    end
end

function iterate_nondiff_y_upper!(tangent_points, λ, lower₁, upper₁, lower₂, upper₂)
    iteration = fill(true, length(lower₁))
    for i in 1:10
        λ[iteration] = clamp.(fslope_y_upper((@view tangent_points[iteration]), (@view lower₁[iteration]), (@view upper₁[iteration]), (@view lower₂[iteration]), (@view upper₂[iteration])),-0.25,0)
        no_iteration = (abs.(λ) .< CUTOFF_SIGMOID_SLOPE)
        iteration = .!no_iteration
        tangent_points[iteration] = solve_∂σ_nondiff_∂y_upper((@view λ[iteration]))
        λ[no_iteration] .= 0
        tangent_points[no_iteration] = lower₂[no_iteration]
    end
end

function iterate_nondiff_y_lower!(tangent_points, λ, lower₁, upper₁, lower₂, upper₂)
    iteration = fill(true, length(lower₁))
    for i in 1:10
        λ[iteration] = clamp.(fslope_y_lower((@view tangent_points[iteration]), (@view lower₁[iteration]), (@view upper₁[iteration]), (@view lower₂[iteration]), (@view upper₂[iteration])),-0.25,0)
        no_iteration = (abs.(λ) .< CUTOFF_SIGMOID_SLOPE)
        iteration = .!no_iteration
        tangent_points[iteration] = solve_∂σ_nondiff_∂y_lower((@view λ[iteration]))
        λ[no_iteration] .= 0
        tangent_points[no_iteration] = upper₂[no_iteration]
    end
end

function fslope_x_upper(tangent_points, lower₁, upper₁, lower₂, upper₂)
   return (σ_nondiff(tangent_points, lower₂) .- σ_nondiff(lower₁, lower₂)) ./ (tangent_points .- lower₁)
end

function fslope_x_lower(tangent_points, lower₁, upper₁, lower₂, upper₂)
    return (σ_nondiff(tangent_points, upper₂) .- σ_nondiff(upper₁, upper₂)) ./ (tangent_points .- upper₁)
end

function fslope_y_upper(tangent_points, lower₁, upper₁, lower₂, upper₂)
    return (σ_nondiff(upper₁, tangent_points) .- σ_nondiff(upper₁, upper₂)) ./ (tangent_points .- upper₂)
end

function fslope_y_lower(tangent_points, lower₁, upper₁, lower₂, upper₂)
    return (σ_nondiff(lower₁, tangent_points) .- σ_nondiff(lower₁, lower₂)) ./ (tangent_points .- lower₂)
end

function set_offset_point!(offset_points, ∂bound, upper₁, lower₁)
    max_mask = ((lower₁ .<= (∂bound ./ 2)) .&& ((∂bound ./ 2) .<= upper₁))
    low_mask = ((∂bound ./ 2) .< lower₁)
    high_mask = (upper₁ .< (∂bound ./ 2))
    offset_points[max_mask] .= (∂bound[max_mask] ./ 2)
    offset_points[low_mask] .= (lower₁[low_mask])
    offset_points[high_mask] .= (upper₁[high_mask])
end

function propagate_layer!(
    ZoutRefVec :: Vector{CachedZonotope},
    Ls :: DiffLayer{
        ONNXSigmoid{S1},
        ONNXSigmoid{S2},
        ONNXSigmoid{S3}},
    inputs :: Vector{DiffZonotope};
    bounds_cache :: Union{Nothing,BoundsCache}=nothing) where {S1, S2, S3}
    #println("Sigmoid Layer")
    @assert length(inputs) == 1 "Sigmoid layer should have exactly one input zonotope"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output zonotope"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]

    @assert !isnothing(bounds_cache)

    # Compute Bounds
    bounds₁ = zono_bounds(Zin.Z₁)
    bounds₂ = zono_bounds(Zin.Z₂)
    ∂bounds = zono_bounds(Zin.∂Z)

    if !bounds_cache.initialized
        bounds_cache.lower₁ = copy(bounds₁[:,1])
        bounds_cache.upper₁ = copy(bounds₁[:,2])
        bounds_cache.lower₂ = copy(bounds₂[:,1])
        bounds_cache.upper₂ = copy(bounds₂[:,2])
        bounds_cache.∂lower = copy(∂bounds[:,1])
        bounds_cache.∂upper = copy(∂bounds[:,2])
        bounds_cache.initialized = true
    else
        bounds_cache.lower₁ .= max.(bounds₁[:,1], bounds_cache.lower₁)
        bounds_cache.upper₁ .= min.(bounds₁[:,2], bounds_cache.upper₁)
        bounds_cache.lower₂ .= max.(bounds₂[:,1], bounds_cache.lower₂)
        bounds_cache.upper₂ .= min.(bounds₂[:,2], bounds_cache.upper₂)
        bounds_cache.∂lower .= max.(∂bounds[:,1], bounds_cache.∂lower)
        bounds_cache.∂upper .= min.(∂bounds[:,2], bounds_cache.∂upper)
    end
    lower₁ = bounds_cache.lower₁
    upper₁ = bounds_cache.upper₁
    lower₂ = bounds_cache.lower₂
    upper₂ = bounds_cache.upper₂
    ∂lower = bounds_cache.∂lower
    ∂upper = bounds_cache.∂upper
    #@info "Bounds Cache: Z₁=[$(lower₁), $(upper₁)], Z₂=[$(lower₂), $(upper₂)], ∂Z=[$(∂lower), $(∂upper)]"

    (
        zero_diff,
        c_all_all,
        all_c_all,
        all_all_neg,
        all_all_pos,
        neg_all_any,
        pos_all_any,
        any_all_any
    ) = get_sigmoid_selectors(bounds₁, bounds₂, ∂bounds)
    # Do NOT use counts created above for new_gen₁ / new_gen₂,
    # because these omit dimensions where difference is still zero
    new_gen₁ = length(lower₁)
    new_gen₂ = length(lower₂)
    ∂new_gen = count(all_all_neg) + count(all_all_pos) + count(neg_all_any) + count(pos_all_any) + count(any_all_any)
    #println("new ∂gen in diff transformers:$(∂new_gen)")
    #println("length only c in diff transformer:$(length(c_all_all)+length(all_c_all))")
    # @debug "Instable Neurons: Network 1: $new_gen₁, Network 2: $new_gen₂, Differential: $∂new_gen"
    Zout_proto = ZoutRef.zonotope_proto # Need this to be able to access the generator ids
    gen_sizes₁ = zeros(Int64,length(Zout_proto.Z₁.generator_ids))
    gen_sizes₂ = zeros(Int64,length(Zout_proto.Z₂.generator_ids))
    ∂gen_sizes = zeros(Int64,length(Zout_proto.∂Z.generator_ids))

    pre_indices_Z₁ = intersect_indices(Zout_proto.Z₁.generator_ids, Zin.Z₁.generator_ids)
    pre_indices_Z₂ = intersect_indices(Zout_proto.Z₂.generator_ids, Zin.Z₂.generator_ids)
    pre_indices₁ = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.Z₁.generator_ids)
    pre_indices₂ = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.Z₂.generator_ids)
    ∂pre_indices = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.∂Z.generator_ids)

    for (i, idx) in enumerate(pre_indices_Z₁)
        gen_sizes₁[idx] = size(Zin.Z₁.Gs[i],2)
    end
    gen_sizes₁[Zout_proto.Z₁.owned_generators] += new_gen₁
    for (i, idx) in enumerate(pre_indices_Z₂)
        gen_sizes₂[idx] = size(Zin.Z₂.Gs[i],2)
    end
    gen_sizes₂[Zout_proto.Z₂.owned_generators] += new_gen₂
    # This mayoverwrite sizes, but columns should be consistent
    # TODO(steuber): Can we make this cleaner?
    for (i, idx) in enumerate(∂pre_indices)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.∂Z.Gs[i],2)) (from ∂Z)"
        ∂gen_sizes[idx] = size(Zin.∂Z.Gs[i],2)
    end
    for (i, idx) in enumerate(pre_indices₁)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.Z₁.Gs[i],2)) (from Z₁)"
        ∂gen_sizes[idx] = size(Zin.Z₁.Gs[i],2)
    end
    for (i, idx) in enumerate(pre_indices₂)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.Z₂.Gs[i],2)) (from Z₂)"
        ∂gen_sizes[idx] = size(Zin.Z₂.Gs[i],2)
    end
    #@info "Generator sizes before new gens: Z₁=$(gen_sizes₁), Z₂=$(gen_sizes₂), ∂Z=$(∂gen_sizes)"
    ∂old_gen = ∂gen_sizes[Zout_proto.∂Z.owned_generators]
    ∂gen_sizes[Zout_proto.∂Z.owned_generators] += ∂new_gen
    # Find idx of generators owned by Z₁ and Z₂ in ∂Z
    idx1 = find_index_position(Zout_proto.∂Z.generator_ids, Zout_proto.Z₁.generator_ids[Zout_proto.Z₁.owned_generators])
    idx2 = find_index_position(Zout_proto.∂Z.generator_ids, Zout_proto.Z₂.generator_ids[Zout_proto.Z₂.owned_generators])
    ∂gen_sizes[idx1] += new_gen₁
    ∂gen_sizes[idx2] += new_gen₂
    Zout_proto = nothing # Avoid missuse
    # @info "Sigmoid DiffZonotope Generators: Z₁=$(gen_sizes₁), Z₂=$(gen_sizes₂), ∂Z=$(∂gen_sizes)"
    Zout = get_zonotope!(ZoutRef, gen_sizes₁, gen_sizes₂, ∂gen_sizes)
    post_indices₁ = intersect_indices(Zout.∂Z.generator_ids, Zout.Z₁.generator_ids)
    post_indices₂ = intersect_indices(Zout.∂Z.generator_ids, Zout.Z₂.generator_ids)

    L1 = get_layer1(Ls)
    L2 = get_layer2(Ls)
    # Compute Zonotopes for individual networks
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁;lower=lower₁, upper=upper₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂;lower=lower₂, upper=upper₂)

    if VeryDiff.USE_DIFFZONO[]
        
        # Reset to zero
        Zout.∂Z.c .= 0.0

        #zero_diff case
        for g in Zout.∂Z.Gs
            g[zero_diff, :] .= 0.0
        end

        μ_all_cases = zeros(length(upper₁))

        #c_all_all case
        updateGeneratorsMul!(Zout.∂Z.Gs, post_indices₂, Zout.Z₂.Gs, (-1.0), c_all_all)
        Zout.∂Z.c[c_all_all] .= .-Zout.Z₂.c[c_all_all] .+ Zout.Z₁.c[c_all_all]

        #all_c_all case
        updateGenerators!(Zout.∂Z.Gs, post_indices₁, Zout.Z₁.Gs, all_c_all)
        Zout.∂Z.c[all_c_all] .= Zout.Z₁.c[all_c_all] .- Zout.Z₂.c[all_c_all]

        # neg_all_any, pos_all_any, any_all_any case
        npa_all_any = (neg_all_any .|| pos_all_any .|| any_all_any)
        dim_npa_all_any = length(@view upper₁[npa_all_any])
        λ_npa_all_any = zeros(dim_npa_all_any)
        ν_npa_all_any = zeros(dim_npa_all_any)
        μ_npa_all_np = zeros(dim_npa_all_any)
        offset_points_upper = zeros(dim_npa_all_any)
        offset_points_lower = zeros(dim_npa_all_any)
        lower₁_npa_all_any = @view lower₁[npa_all_any]
        upper₁_npa_all_any = @view upper₁[npa_all_any]
        ∂lower_npa_all_any = @view ∂lower[npa_all_any]
        ∂upper_npa_all_any = @view ∂upper[npa_all_any]
        set_offset_point!(offset_points_upper, ∂upper_npa_all_any, upper₁_npa_all_any, lower₁_npa_all_any)
        set_offset_point!(offset_points_lower, ∂lower_npa_all_any, upper₁_npa_all_any, lower₁_npa_all_any)
        upper_deriv = ∂σ_diff_∂y(offset_points_upper, ∂upper_npa_all_any)
        lower_deriv = ∂σ_diff_∂y(offset_points_lower, ∂lower_npa_all_any)
        cutoff_upper = σ_diff(offset_points_upper, ∂upper_npa_all_any) ./ (∂upper_npa_all_any .- ∂lower_npa_all_any)
        cutoff_lower = .-σ_diff(offset_points_lower, ∂lower_npa_all_any) ./ (∂upper_npa_all_any .- ∂lower_npa_all_any)
        cutoff = ifelse.(cutoff_lower .< cutoff_upper, cutoff_lower, cutoff_upper)
        slope = ifelse.(lower_deriv .< upper_deriv, lower_deriv, upper_deriv)
        λ_npa_all_any = ifelse.(cutoff .< slope, cutoff, slope)
        μ_npa_all_np = 0.5 .* (.-λ_npa_all_any .* ∂upper_npa_all_any .+ σ_diff(offset_points_upper, ∂upper_npa_all_any)
        .+λ_npa_all_any .* ∂lower_npa_all_any .- σ_diff(offset_points_lower, ∂lower_npa_all_any))
        ν_npa_all_any = 0.5 .* (.-λ_npa_all_any .* ∂upper_npa_all_any .+ σ_diff(offset_points_upper, ∂upper_npa_all_any)
        .-λ_npa_all_any .* ∂lower_npa_all_any .+ σ_diff(offset_points_lower, ∂lower_npa_all_any))

        μ_all_cases[npa_all_any] .= μ_npa_all_np

        updateGeneratorsMul!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, λ_npa_all_any, npa_all_any) 
        Zout.∂Z.c[npa_all_any] .= (λ_npa_all_any .* (@view Zin.∂Z.c[npa_all_any])) .+ ν_npa_all_any


        # all_all_neg or all_all_pos case
        all_all_np = (all_all_neg .|| all_all_pos)
        dim_all_all_np = length((@view upper₁[all_all_np]))
        λ_x_all_all_np = zeros(dim_all_all_np)
        λ_y_all_all_np = zeros(dim_all_all_np)
        ν_all_all_np = zeros(dim_all_all_np)
        μ_all_all_np = zeros(dim_all_all_np)

        let 
            upper₁_all_all_np = @view upper₁[all_all_np]
            lower₁_all_all_np = @view lower₁[all_all_np]
            upper₂_all_all_np = @view upper₂[all_all_np]
            lower₂_all_all_np = @view lower₂[all_all_np]

            λ_x = zeros(dim_all_all_np)
            λ_y = zeros(dim_all_all_np)

            λ_x_upper = zeros(dim_all_all_np)
            λ_x_lower = zeros(dim_all_all_np)
            λ_y_upper = zeros(dim_all_all_np)
            λ_y_lower = zeros(dim_all_all_np)

            tangent_points_x_upper = copy(upper₁_all_all_np)
            tangent_points_x_lower = copy(lower₁_all_all_np)
            tangent_points_y_upper = copy(lower₂_all_all_np)
            tangent_points_y_lower = copy(upper₂_all_all_np)

            intial_derivatives_x_upper = ∂σ_nondiff_∂x(upper₁_all_all_np, lower₂_all_all_np)
            intial_derivatives_x_lower = ∂σ_nondiff_∂x(lower₁_all_all_np, upper₂_all_all_np)
            intial_derivatives_y_upper = ∂σ_nondiff_∂y(upper₁_all_all_np, lower₂_all_all_np)
            intial_derivatives_y_lower = ∂σ_nondiff_∂y(lower₁_all_all_np, upper₂_all_all_np)

            secant_slope_x_upper = fslope_x_upper(upper₁_all_all_np, lower₁_all_all_np, upper₁_all_all_np, lower₂_all_all_np, upper₂_all_all_np)
            secant_slope_x_lower = fslope_x_lower(lower₁_all_all_np, lower₁_all_all_np, upper₁_all_all_np, lower₂_all_all_np, upper₂_all_all_np)
            secant_slope_y_upper = fslope_y_upper(lower₂_all_all_np, lower₁_all_all_np, upper₁_all_all_np, lower₂_all_all_np, upper₂_all_all_np)
            secant_slope_y_lower = fslope_y_lower(upper₂_all_all_np, lower₁_all_all_np, upper₁_all_all_np, lower₂_all_all_np, upper₂_all_all_np)

            use_secant_x_upper = (intial_derivatives_x_upper .>= secant_slope_x_upper)
            use_secant_x_lower = (intial_derivatives_x_lower .>= secant_slope_x_lower)
            use_secant_y_upper = (intial_derivatives_y_upper .<= secant_slope_y_upper)
            use_secant_y_lower = (intial_derivatives_y_lower .<= secant_slope_y_lower)

            need_iteration_x_upper = .!use_secant_x_upper
            need_iteration_x_lower = .!use_secant_x_lower
            need_iteration_y_upper = .!use_secant_y_upper
            need_iteration_y_lower = .!use_secant_y_lower

            λ_x_upper[use_secant_x_upper] = secant_slope_x_upper[use_secant_x_upper]
            λ_x_lower[use_secant_x_lower] = secant_slope_x_lower[use_secant_x_lower]
            λ_y_upper[use_secant_y_upper] = secant_slope_y_upper[use_secant_y_upper]
            λ_y_lower[use_secant_y_lower] = secant_slope_y_lower[use_secant_y_lower]

            iterate_nondiff_x_upper!((@view tangent_points_x_upper[need_iteration_x_upper]), (@view λ_x_upper[need_iteration_x_upper]), (@view lower₁_all_all_np[need_iteration_x_upper]),
            (@view upper₁_all_all_np[need_iteration_x_upper]), (@view lower₂_all_all_np[need_iteration_x_upper]), (@view upper₂_all_all_np[need_iteration_x_upper]))
            iterate_nondiff_x_lower!((@view tangent_points_x_lower[need_iteration_x_lower]), (@view λ_x_lower[need_iteration_x_lower]), (@view lower₁_all_all_np[need_iteration_x_lower]),
            (@view upper₁_all_all_np[need_iteration_x_lower]), (@view lower₂_all_all_np[need_iteration_x_lower]), (@view upper₂_all_all_np[need_iteration_x_lower]))
            iterate_nondiff_y_upper!((@view tangent_points_y_upper[need_iteration_y_upper]), (@view λ_y_upper[need_iteration_y_upper]), (@view lower₁_all_all_np[need_iteration_y_upper]),
            (@view upper₁_all_all_np[need_iteration_y_upper]), (@view lower₂_all_all_np[need_iteration_y_upper]), (@view upper₂_all_all_np[need_iteration_y_upper]))
            iterate_nondiff_y_lower!((@view tangent_points_y_lower[need_iteration_y_lower]), (@view λ_y_lower[need_iteration_y_lower]), (@view lower₁_all_all_np[need_iteration_y_lower]), 
            (@view upper₁_all_all_np[need_iteration_y_lower]), (@view lower₂_all_all_np[need_iteration_y_lower]), (@view upper₂_all_all_np[need_iteration_y_lower]))

            use_upper_x_slope = (λ_x_upper .<= λ_x_lower)
            use_upper_y_slope = (λ_y_upper .>= λ_y_lower)
            use_lower_x_slope = .!use_upper_x_slope
            use_lower_y_slope = .!use_upper_y_slope

            λ_x[use_upper_x_slope] = λ_x_upper[use_upper_x_slope]
            λ_x[use_lower_x_slope] = λ_x_lower[use_lower_x_slope]
            λ_y[use_upper_y_slope] = λ_y_upper[use_upper_y_slope]
            λ_y[use_lower_y_slope] = λ_y_lower[use_lower_y_slope]

            λ_x_zero = (λ_x .== 0)
            λ_y_zero = (λ_y .== 0)

            tangent_points_x_upper[use_lower_x_slope .&& λ_x_zero] = upper₁_all_all_np[use_lower_x_slope .&& λ_x_zero]
            tangent_points_x_upper[use_lower_x_slope .&& .!λ_x_zero] = solve_∂σ_nondiff_∂x_upper((@view λ_x[use_lower_x_slope .&& .!λ_x_zero])) 
            tangent_points_x_lower[use_upper_x_slope .&& λ_x_zero] = lower₁_all_all_np[use_upper_x_slope .&& λ_x_zero]
            tangent_points_x_lower[use_upper_x_slope .&& .!λ_x_zero] = solve_∂σ_nondiff_∂x_lower((@view λ_x[use_upper_x_slope .&& .!λ_x_zero]))
            
            tangent_points_y_upper[use_lower_y_slope .&& λ_y_zero] = lower₂_all_all_np[use_lower_y_slope .&& λ_y_zero]
            tangent_points_y_upper[use_lower_y_slope .&& .!λ_y_zero] = solve_∂σ_nondiff_∂y_upper((@view λ_y[use_lower_y_slope .&& .!λ_y_zero]))
            tangent_points_y_lower[use_upper_y_slope .&& λ_y_zero] = upper₂_all_all_np[use_upper_y_slope .&& λ_y_zero]
            tangent_points_y_lower[use_upper_y_slope .&& .!λ_y_zero] = solve_∂σ_nondiff_∂y_lower((@view λ_y[use_upper_y_slope .&& .!λ_y_zero]))
            
            λ_x_all_all_np .= λ_x
            λ_y_all_all_np .= λ_y

            ν_all_all_np .= 0.5 .* (.-λ_x .* tangent_points_x_lower .-λ_y .* tangent_points_y_lower .+ σ_nondiff(tangent_points_x_lower, tangent_points_y_lower) 
            .- λ_x .* tangent_points_x_upper .- λ_y .* tangent_points_y_upper .+ σ_nondiff(tangent_points_x_upper, tangent_points_y_upper))
            μ_all_all_np .= 0.5 .* (λ_x .* tangent_points_x_lower .+ λ_y .* tangent_points_y_lower .- σ_nondiff(tangent_points_x_lower, tangent_points_y_lower) 
            .- λ_x .* tangent_points_x_upper .- λ_y .* tangent_points_y_upper .+ σ_nondiff(tangent_points_x_upper, tangent_points_y_upper))

        end

        updateGeneratorsMul!(Zout.∂Z.Gs, pre_indices₁, Zin.Z₁.Gs, λ_x_all_all_np, all_all_np)
        updateGeneratorsAddMul!(Zout.∂Z.Gs, pre_indices₂, Zin.Z₂.Gs, λ_y_all_all_np, all_all_np)
        Zout.∂Z.c[all_all_np] .= (λ_x_all_all_np .* (@view Zin.Z₁.c[all_all_np])) .+ (λ_y_all_all_np .* (@view Zin.Z₂.c[all_all_np])) .+ ν_all_all_np


        μ_all_cases[all_all_np] .= μ_all_all_np

        # Add new generators from c
        dim = length(any_all_any)
        c_pos = findall(@simd_bool_expr dim (all_all_neg | all_all_pos | neg_all_any | pos_all_any | any_all_any ))
        A = Zout.∂Z.Gs[Zout.∂Z.owned_generators]
        @inbounds for i in 1:length(c_pos)
            row = c_pos[i]
            col = ∂old_gen + i
            if all_all_neg[row]
                A[row, col] = μ_all_cases[row]
            elseif all_all_pos[row]
                A[row, col] = μ_all_cases[row]
            elseif neg_all_any[row]
                A[row, col] = μ_all_cases[row]
            elseif pos_all_any[row]
                A[row, col] = μ_all_cases[row]
            elseif any_all_any[row]
                A[row, col] = μ_all_cases[row]
            end
        end
    end
end