function get_top1_both_confident_property(delta1, delta2;naive=false)
    dist1 = iszero(delta1) ? 0.0 : log(delta1/(1-delta1))
    dist2 = iszero(delta2) ? 0.0 : log(delta2/(1-delta2))

    return (N1, N2, Zin, Zout, verification_status) -> begin
        global TOP1_FOUND_CONCRETE_DELTA
        if VeryDiff.FIRST_ROUND[]
            TOP1_FOUND_CONCRETE_DELTA[] = false
        end
        if isnothing(verification_status)
            verification_status = Dict{Tuple{Int,Int},Bool}()
        end
        input_dim = length(Zin.Z₁.c)
        res1 = N1(Zin.Z₁.c)
        res2 = N2(Zin.Z₂.c)
        argmax_N1 = argmax(res1)
        argmax_N2 = argmax(res2)
        softmax_N1 = exp.(res1)/sum(exp.(res1))
        if argmax_N1 != argmax_N2
            if softmax_N1[argmax_N1] >= delta1
                println("Found cex")
                println("N1 Probability: $(softmax_N1[argmax_N1]) >= $delta1")
                return false, (Zin.Z₁.c, (argmax_N1, argmax_N2)), nothing, nothing, 0.0
            else
                second_largest = sort(res1,rev=true)[2]
                if !iszero(delta1) && res1[argmax_N1]-second_largest >= dist1
                    println("Found spurious cex")
                    println("N1 Probability: $(softmax_N1[argmax_N1]) < $delta1")
                    println("but difference $(res1[argmax_N1]-second_largest) >= $dist1 (approximate bound)")
                end
            end
        end
        property_satisfied = true
        distance_bound = 0.0
        any_feasible = false
        common_generator_indices = union(
            Zout.Z₁.generator_ids,
            union(Zout.Z₂.generator_ids, Zout.∂Z.generator_ids))
        variable_offsets = Int64[1]
        for id in common_generator_indices
            for curZ in [Zout.∂Z, Zout.Z₁, Zout.Z₂]
                id_pos = attempt_find_index_position(curZ.generator_ids, id)
                if id_pos > 0
                    next_offset = variable_offsets[end] + size(curZ.Gs[id_pos], 2)
                    push!(variable_offsets, next_offset)
                    break
                end
            end
        end
        indices₁ = intersect_indices(common_generator_indices, Zout.Z₁.generator_ids)
        indices₂ = intersect_indices(common_generator_indices, Zout.Z₂.generator_ids)
        ∂indices = intersect_indices(common_generator_indices, Zout.∂Z.generator_ids)

        in_indices₁ = intersect_indices(common_generator_indices, Zin.Z₁.generator_ids)
        in_indices₂ = intersect_indices(common_generator_indices, Zin.Z₂.generator_ids)
        output_dim = length(Zout.Z₁.c)
        for top_index in 1:output_dim
            if USE_GUROBI[]
                model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
            else
                model = Model(GLPK.Optimizer)
            end
            set_time_limit_sec(model, 10)
            var_num = variable_offsets[end]-1
            @variable(model,-1.0 <= x[1:var_num] <= 1.0)
            
            # Constraint 1: Maximal output of first network is top_index
            offset1_start = variable_offsets[indices₁[1]]
            offset1_end = variable_offsets[indices₁[1]+1] - 1
            lhs = ((@view Zout.Z₁.Gs[1][1:end .!= top_index,:]) .- (@view Zout.Z₁.Gs[1][top_index:top_index,:]))*x[offset1_start:offset1_end]
            rhs = (Zout.Z₁.c[top_index] .- (@view Zout.Z₁.c[1:end .!= top_index])) .- dist1
            
            for i in 2:length(Zout.Z₁.Gs)
                curG = Zout.Z₁.Gs[i]
                curIdx = indices₁[i]
                offset_start = variable_offsets[curIdx]
                offset_end = variable_offsets[curIdx + 1] - 1
                lhs .+= ((@view curG[1:end .!= top_index, :]) .- (@view curG[top_index:top_index,:])) * x[offset_start:offset_end]
            end
            @constraint(model, lhs .<= rhs)

            if !naive
                # Constraint 2:
                # Output difference between networks is given by the differential zonotope (∂Z)
                #       ∂Z - (Z₁ - Z₂) = 0
                # <->   (∂G + ∂c) - ((G₁ + c₁) - (G₂ + c₂)) = 0
                # <->   ∂G + ∂c - G₁ - c₁ + G₂ + c₂ = 0
                # <->   ∂G - G₁ + G₂ = c₁ - c₂ - ∂c
                G2 = zeros(output_dim, var_num)
                for (i, curIdx) in enumerate(∂indices)
                    offset_start = variable_offsets[curIdx]
                    offset_end = variable_offsets[curIdx + 1] - 1
                    G2[:,offset_start:offset_end] .= Zout.∂Z.Gs[i]
                end

                for (i, curIdx) in enumerate(indices₁)
                    offset_start = variable_offsets[curIdx]
                    offset_end = variable_offsets[curIdx + 1] - 1
                    G2[:,offset_start:offset_end] .-= Zout.Z₁.Gs[i]
                end

                for (i, curIdx) in enumerate(indices₂)
                    offset_start = variable_offsets[curIdx]
                    offset_end = variable_offsets[curIdx + 1] - 1
                    G2[:, offset_start:offset_end] .+= Zout.Z₂.Gs[i]
                end
                @constraint(model,
                    G2*x .== (Zout.Z₁.c .- Zout.∂Z.c .- Zout.Z₂.c)
                )
            end

            # Check if model is feasible
            @objective(model,Max,0)
            optimize!(model)
            
            if termination_status(model) == MOI.INFEASIBLE
                # Model is infeasible -> top_index is never maximal with delta
                for other_index in 1:output_dim
                    verification_status[(top_index,other_index)]=true
                end
            else
                # Model is feasible -> top_index can be maximal in NN1 with delta
                # Now check that all output dimensions of NN2 (other than top_index)
                # are less than the output at top_index

                # ...but before we do that:
                # Check if we found concrete evidence for feasibility of confidence delta
                if !TOP1_FOUND_CONCRETE_DELTA[]
                    input1 = copy(Zin.Z₁.c)
                    for (i, curIdx) in enumerate(in_indices₁)
                        offset_start = variable_offsets[curIdx]
                        offset_end = offset_start + size(Zin.Z₁.Gs[i],2) - 1
                        input1 .+= Zin.Z₁.Gs[i] * value.(x[offset_start:offset_end])
                    end
                    # input2 = copy(Zin.Z₂.c)
                    # for (i, curIdx) in enumerate(in_indices₂)
                    #     offset_start = variable_offsets[curIdx]
                    #     offset_end = offset_start + size(Zin.Z₂.Gs[i],2)
                    #     input2 .+= Zin.Z₂.Gs[i] * value.(x[offset_start:offset_end])
                    # end
                    res1 = N1(input1)
                    argmax_N1 = argmax(res1)
                    softmax_N1 = exp.(res1)/sum(exp.(res1))
                    if softmax_N1[argmax_N1] >= delta1
                        println("[TOP-1] required confidence ($(softmax_N1[argmax_N1])≥$delta1) is feasible for index $argmax_N1")
                        TOP1_FOUND_CONCRETE_DELTA[]=true
                    else
                        #println("[TOP-1] did not find required confidence yet.")
                    end
                end
                # ...also before we do that:
                # Record that we found at least one feasible top_index
                any_feasible = true
                
                # ...now we'll check the output dimensions of NN2:
                for other_index in 1:output_dim
                    if other_index != top_index && !haskey(verification_status, (top_index,other_index))
                        # We are indeed considering an output index other than top_index
                        # This other index has also not yet been verified

                        # Set up objective:
                        # Maximize gap between other_index and top_index in NN2
                        a = zeros(var_num)

                        for (i, curIdx) in enumerate(indices₂)
                            offset_start = variable_offsets[curIdx]
                            offset_end = variable_offsets[curIdx + 1] - 1
                            a[offset_start:offset_end] .= Zout.Z₂.Gs[i][other_index,:] .- Zout.Z₂.Gs[i][top_index,:]
                        end
                        # What needs to be satisfied for this to be a violation:
                        # ∀ i. other_index≠i ⟶ out[other_index] - out[i] >= dist2
                        # ->
                        # out[other_index] - out[top_index] >= dist2
                        # Thus:
                        # (out[other_index] - out[top_index]) < dist2
                        # ->
                        # ¬(∀ i. other_index≠i ⟶ out[other_index] - out[i] >= dist2)
                        @objective(model,Max,a'*x + Zout.Z₂.c[other_index] - Zout.Z₂.c[top_index])
                        
                        # Compute threshold for objective
                        threshold = dist2

                        # If the optimal value is < threshold, then the property is satisfied
                        # (it is impossible for other_index to be greater than top_index)
                        # otherwise (optimal >= threshold) we may have found a counterexample

                        if USE_GUROBI[] # we are using GUROBI -> set objective/bound thresholds
                            set_optimizer_attribute(model, "Cutoff", threshold-1e-4)
                        end
                        optimize!(model)

                        # TODO(steuber): What to do with this result?

                        model_status = termination_status(model)
                        # Model MUST be feasible since we did not add any constraints
                        @assert model_status != MOI.INFEASIBLE

                        # Model should be optimal or have reached the objective limit
                        # any other status -> split and retry
                        if model_status != MOI.OPTIMAL && model_status != MOI.OBJECTIVE_LIMIT
                            println("[GUROBI] Irregular model status: $model_status")
                            property_satisfied = false
                            # if has_values(model)
                            #     distance_bound = max(distance_bound, objective_value(model))
                            # end
                            continue
                        end
                        
                        # LP has reached optimization limit
                        # other_index cannot be maximal with this confidence
                        if model_status == MOI.OBJECTIVE_LIMIT || objective_value(model) < threshold
                            verification_status[(top_index,other_index)]=true
                        else
                            new_model, reference_map = copy_model(model)
                            set_optimizer(new_model, USE_GUROBI[] ? () -> Gurobi.Optimizer(GRB_ENV[]) : GLPK.Optimizer)
                            x_new = reference_map[x]
                            # Constraint 3: Maximal output of second network is other_index
                            offset2_start = variable_offsets[indices₂[1]]
                            offset2_end = variable_offsets[indices₂[1]+1] - 1
                            lhs2 = ((@view Zout.Z₂.Gs[1][1:end .!= other_index,:]) .- (@view Zout.Z₂.Gs[1][other_index:other_index,:]))*x_new[offset2_start:offset2_end]
                            rhs2 = (Zout.Z₂.c[other_index] .- (@view Zout.Z₂.c[1:end .!= other_index])) .- dist2

                            obj = Zout.Z₂.Gs[1][other_index,:]' * x_new[offset2_start:offset2_end]

                            for i in 2:length(Zout.Z₂.Gs)
                                curG = Zout.Z₂.Gs[i]
                                curIdx = indices₂[i]
                                offset_start = variable_offsets[curIdx]
                                offset_end = variable_offsets[curIdx + 1] - 1
                                lhs2 .+= ((@view curG[1:end .!= other_index, :]) .- (@view curG[other_index:other_index,:])) * x_new[offset_start:offset_end]
                                obj += curG[other_index,:]' * x_new[offset_start:offset_end]
                            end
                            @constraint(new_model, lhs2 .<= rhs2)

                            @objective(new_model,Max,obj)

                            optimize!(new_model)

                            model_status = termination_status(new_model)
                            if model_status == MOI.INFEASIBLE
                                # Model is infeasible -> other_index cannot be maximal with delta2
                                verification_status[(top_index,other_index)]=true
                            else
                                # Potentially we found a counterexample
                                # -> check that
                                # distance_bound = max(distance_bound, objective_value(new_model))
                                input1 = copy(Zin.Z₁.c)
                                for (i, curIdx) in enumerate(in_indices₁)
                                    offset_start = variable_offsets[curIdx]
                                    offset_end = offset_start + size(Zin.Z₁.Gs[i],2) - 1
                                    input1 .+= Zin.Z₁.Gs[i] * value.(x_new[offset_start:offset_end])
                                end
                                # Important: Use input generated by Z₂ here
                                # Important due to robustness queries where input Z₁ and Z₂ differ
                                input2 = copy(Zin.Z₂.c)
                                for (i, curIdx) in enumerate(in_indices₂)
                                    offset_start = variable_offsets[curIdx]
                                    offset_end = offset_start + size(Zin.Z₂.Gs[i],2) - 1
                                    input2 .+= Zin.Z₂.Gs[i] * value.(x_new[offset_start:offset_end])
                                end
                                res1 = N1(input1)
                                res2 = N2(input2)
                                argmax_N1 = argmax(res1)
                                argmax_N2 = argmax(res2)
                                softmax_N1 = exp.(res1)/sum(exp.(res1))
                                softmax_N2 = exp.(res2)/sum(exp.(res2))
                                if argmax_N1 != argmax_N2
                                    # N1 and N2 indeed differ in their classification for input
                                    # But does N1 have enough confidence?
                                    if softmax_N1[argmax_N1] >= delta1 && softmax_N2[argmax_N2] >= delta2
                                        # N1 and N2 have sufficient confidence -> concrete counterexample
                                        println("Found cex")
                                        second_most = sort(softmax_N1,rev=true)[2]
                                        println("N1 ($argmax_N1): $(softmax_N1[argmax_N1]) (vs. $second_most)")
                                        println("N2 ($argmax_N2): $(softmax_N2[argmax_N2])")
                                        println("N1 Probability: $(softmax_N1[argmax_N1]) >= $delta1")
                                        println("N2 Probability: $(softmax_N2[argmax_N2]) >= $delta2")
                                        return false, (input1, (argmax_N1, argmax_N2)), nothing, nothing, 0.0
                                    else
                                        # N1 or N2 do not have enough confidence
                                        second_largest1 = sort(res1,rev=true)[2]
                                        second_largest2 = sort(res2,rev=true)[2]
                                        if res1[argmax_N1]-second_largest1 >= dist1 && res2[argmax_N2]-second_largest2 >= dist2
                                            println("Found spurious cex (N1 is the problem)")
                                            println("N1 Probability: $(softmax_N1[argmax_N1]) < $delta1")
                                            println("but difference $(res1[argmax_N1]-second_largest1) >= $dist1 (approximate bound)")
                                            println("N2 Probability: $(softmax_N2[argmax_N2]) < $delta2")
                                            println("but difference $(res2[argmax_N2]-second_largest2) >= $dist2 (approximate bound)")
                                        end
                                        property_satisfied = false
                                    end
                                else
                                    # Counterexample was spurious in the sense that N1 and N2
                                    # have the same maximum
                                    property_satisfied = false
                                end
                            end
                        end
                    end
                end
            end
            # generator_importance .*= sum(abs,Zin.Z₁.G,dims=1)[1,:]
        end
        # if property_satisfied
        #     println("Zonotope Top 1 Equivalent!")
        # end
        return property_satisfied, nothing, nothing, verification_status, distance_bound
    end
end