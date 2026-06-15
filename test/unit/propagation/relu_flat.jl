@testset "Sampled Points Within Output Bounds" begin
        input_dim = 10
        num_samples = 20_000
        tolerance = 1e-4
        output_dim = 10

        for runs in 1:100
            failed = false
            mask_failed = falses(output_dim)
            for depth in 1:1 #1:15
                # Create layer dimensions (same for both networks)
                println("depth of tested Network: $(depth)")
                layer_dims = [rand(output_dim:output_dim) for _ in 1:depth]
                #push!(layer_dims, 10)  # Output dimension is 10
                
                @debug "Creating networks..."
                #println("Creating networks...")
                # Create networks with same structure
                N1, N2 = make_dense_pair(input_dim, layer_dims;relu=true)
                depth = length(layer_dims)

                # Create input bounds [-1, 1]^input_dim
                low = fill(-1.0, input_dim)
                high = fill(1.0, input_dim)
                # Sample points from input space and propagate through both networks
                input_samples = sample_points_in_hypercube(low, high, num_samples)

                N_gemini = GeminiNetwork(N1, N2)

                N1 = executable_network(N1)
                N2 = executable_network(N2)
                
                # Create VerificationTask
                verification_task = create_verification_task(low, high)
                
                # Propagate zonotope through network
                prop_state = PropState(true)
                prepare_prop_state!(prop_state, verification_task)
                prop_state = propagate!(N_gemini, prop_state)
                @debug "Propagation through Gemini Network complete."
                @debug "$(length(prop_state.zono_storage.zonotopes)) zonotopes in storage."
                Zout = prop_state.zono_storage.zonotopes[end].zonotope
                println("prop_state.zono_storage.zonotopes length should be equal double the depth: $(length(prop_state.zono_storage.zonotopes))")
                
                # Get output bounds from zonotope
                bounds_z1 = zono_bounds(Zout.Z₁)
                bounds_z2 = zono_bounds(Zout.Z₂)
                bounds_z∂ = zono_bounds(Zout.∂Z)
                # @info "Output bounds for Network 1: $bounds_z1"
                # @info "Output bounds for Network 2: $bounds_z2"
                
                n1_violations = 0
                n2_violations = 0
                diff_violations = 0
                for i in 1:num_samples
                    x = input_samples[:, i]

                    Zin = prop_state.zono_storage.zonotopes[1].zonotope.Z₁
                    @assert Zin.c .+ Zin.Gs[1]*x ≈ x atol=1e-5
                    Zin2 = prop_state.zono_storage.zonotopes[1].zonotope.Z₂
                    @assert Zin2.c .+ Zin2.Gs[1]*x ≈ x atol=1e-5
                    
                    # Propagate through N1
                    y1 = N1(x)
                    
                    # Propagate through N2
                    y2 = N2(x)

                    # Now we may have additional constraints so we must check Zonotope containment
                    # Sum up additional generators
                    Z1_range = sum(g->sum(abs, g, dims=2),Zout.Z₁.Gs[2:end];init=zeros(size(Zout.Z₁.c)))
                    Z2_range = sum(g->sum(abs, g, dims=2),Zout.Z₂.Gs[2:end];init=zeros(size(Zout.Z₂.c)))
                    input_component = Zout.Z₁.c .+ Zout.Z₁.Gs[1]*x
                    @test all(input_component .- Z1_range .<= y1 .+ 1e-5)
                    @test all(y1 .<= input_component .+ Z1_range .+ 1e-5)
                    if !all(input_component .- Z1_range .<= y1 .+ 1e-5) || !all(y1 .<= input_component .+ Z1_range .+ 1e-5)
                        @info "Output: $y1"
                        @info "Zonotope bounds (agnostic): $(bounds_z1)"
                        @info "Zonotope bounds (generator sum): $((input_component .- Z1_range, input_component .+ Z1_range))"
                        return
                    end
                    input_component2 = Zout.Z₂.c .+ Zout.Z₂.Gs[1]*x
                    @test all(input_component2 .- Z2_range .<= y2 .+ 1e-5)
                    @test all(y2 .<= input_component2 .+ Z2_range .+ 1e-5)
                    if !all(input_component2 .- Z2_range .<= y2 .+ 1e-5) || !all(y2 .<= input_component2 .+ Z2_range .+ 1e-5)
                        @info "Output: $y2"
                        @info "Zonotope bounds (agnostic): $(bounds_z2)"
                        @info "Zonotope bounds (generator sum): $((input_component2 .- Z2_range, input_component2 .+ Z2_range))"
                        return
                    end
                    # Difference Zonotope
                    diff_range = sum(g->sum(abs, g, dims=2),Zout.∂Z.Gs[2:end];init=zeros(size(Zout.∂Z.c)))
                    if length(Zout.∂Z.Gs) >= 1
                        input_component_diff = Zout.∂Z.c .+ Zout.∂Z.Gs[1]*x
                    else
                        input_component_diff = Zout.∂Z.c
                    end
                    @test all((input_component_diff .- diff_range) .<= ((y1 .- y2) .+ 1e-5))
                    @test all((y1 .- y2) .<= ((input_component_diff .+ diff_range) .+ 1e-5))
                    if !all((input_component_diff .- diff_range) .<= ((y1 .- y2) .+ 1e-5)) || !all((y1 .- y2) .<= ((input_component_diff .+ diff_range) .+ 1e-5))
                        failed = true
                        mask_failed = (.!((input_component_diff .- diff_range) .<= ((y1 .- y2) .+ 1e-5)) .|| .!((y1 .- y2) .<= ((input_component_diff .+ diff_range) .+ 1e-5)))
                        mask_failed = mask_failed[:,1]
                    end
                    # Check if outputs are within bounds
                    for d in 1:output_dim
                        if y1[d] < bounds_z1[d, 1] - tolerance || y1[d] > bounds_z1[d, 2] + tolerance
                            n1_violations += 1
                        end
                        
                        if y2[d] < bounds_z2[d, 1] - tolerance || y2[d] > bounds_z2[d, 2] + tolerance
                            n2_violations += 1
                        end
                        if (y1[d] - y2[d]) < bounds_z∂[d, 1] - tolerance || (y1[d] - y2[d]) > bounds_z∂[d, 2] + tolerance
                            diff_violations += 1
                        end
                    end
                end
                if failed
                    input_zonotopes = prop_state.zono_storage.zonotopes[1].zonotope
                    affine_zonotopes = prop_state.zono_storage.zonotopes[2].zonotope
                    output_zonotopes = prop_state.zono_storage.zonotopes[3].zonotope
                    println("input Z1 [lower bounds, upper bounds]: $([zono_bounds(input_zonotopes.Z₁)[mask_failed,1], zono_bounds(input_zonotopes.Z₁)[mask_failed,2]])")
                    println("input Z2 [lower bounds, upper bounds]: $([zono_bounds(input_zonotopes.Z₂)[mask_failed,1], zono_bounds(input_zonotopes.Z₂)[mask_failed,2]])")
                    println("input ∂Z [lower bounds, upper bounds]: $([zono_bounds(input_zonotopes.∂Z)[mask_failed,1], zono_bounds(input_zonotopes.∂Z)[mask_failed,2]])")
                    println("affine Z1 [lower bounds, upper bounds]: $([zono_bounds(affine_zonotopes.Z₁)[mask_failed,1], zono_bounds(affine_zonotopes.Z₁)[mask_failed,2]])")
                    println("affine Z2 [lower bounds, upper bounds]: $([zono_bounds(affine_zonotopes.Z₂)[mask_failed,1], zono_bounds(affine_zonotopes.Z₂)[mask_failed,2]])")
                    println("affine ∂Z [lower bounds, upper bounds]: $([zono_bounds(affine_zonotopes.∂Z)[mask_failed,1], zono_bounds(affine_zonotopes.∂Z)[mask_failed,2]])")
                    println("output Z1 [lower bounds, upper bounds]: $([zono_bounds(output_zonotopes.Z₁)[mask_failed,1], zono_bounds(output_zonotopes.Z₁)[mask_failed,2]])")
                    println("output Z2 [lower bounds, upper bounds]: $([zono_bounds(output_zonotopes.Z₂)[mask_failed,1], zono_bounds(output_zonotopes.Z₂)[mask_failed,2]])")
                    println("output ∂Z [lower bounds, upper bounds]: $([zono_bounds(output_zonotopes.∂Z)[mask_failed,1], zono_bounds(output_zonotopes.∂Z)[mask_failed,2]])")
                    bounds_affine_Z1 = zono_bounds(affine_zonotopes.Z₁)
                    bounds_affine_Z2 = zono_bounds(affine_zonotopes.Z₂)
                    bounds_affine_∂Z = zono_bounds(affine_zonotopes.∂Z)
                    (
                    zero_diff,
                    c_all_all,
                    all_c_all,
                    all_all_neg,
                    all_all_pos,
                    neg_all_any,
                    pos_all_any,
                    any_all_any
                    ) = get_sigmoid_selectors(bounds_affine_Z1, bounds_affine_Z2, bounds_affine_∂Z)
                    println("c_all_all: $(c_all_all[mask_failed])")
                    println("all_c_all: $(all_c_all[mask_failed])")
                    println("affine Z1.c: $(affine_zonotopes.Z₁.c[mask_failed])")
                    println("affine Z2.c: $(affine_zonotopes.Z₂.c[mask_failed])")
                    println("affine ∂Z.c :$(affine_zonotopes.∂Z.c[mask_failed])")
                    println("output Z1.c: $(output_zonotopes.Z₁.c[mask_failed])")
                    println("output Z2.c: $(output_zonotopes.Z₂.c[mask_failed])")
                    println("output ∂Z.c :$(output_zonotopes.∂Z.c[mask_failed])")
                    println("affine Z1.Gs: $(affine_zonotopes.Z₁.Gs[1][mask_failed,:])")
                    println("affine Z2.Gs: $(affine_zonotopes.Z₂.Gs[1][mask_failed,:])")
                    println("affine ∂Z.Gs :$(affine_zonotopes.∂Z.Gs[1][mask_failed,:])")
                    println("output Z1.Gs: $(output_zonotopes.Z₁.Gs[1][mask_failed,:])")
                    println("output Z2.Gs: $(output_zonotopes.Z₂.Gs[1][mask_failed,:])")
                    println("output ∂Z.Gs :$(output_zonotopes.∂Z.Gs[1][mask_failed,:])")
                    println("all affine Z1.Gs: $(affine_zonotopes.Z₁.Gs)")
                    println("all affine Z2.Gs: $(affine_zonotopes.Z₂.Gs)")
                    println("all affine ∂Z.Gs :$(affine_zonotopes.∂Z.Gs)")
                    println("all output Z1.Gs: $(output_zonotopes.Z₁.Gs)")
                    println("all output Z2.Gs: $(output_zonotopes.Z₂.Gs)")
                    println("all output ∂Z.Gs :$(output_zonotopes.∂Z.Gs)")

                end
                violations = n1_violations + n2_violations + diff_violations
                @info "Found $violations bound violations out of $(num_samples * 30) total checks"
                if violations > 0
                    @info "Of which N1: $n1_violations"
                    @info "Of which N2: $n2_violations"
                    @info "Of which Diff: $diff_violations"
                end
                # All sampled points should be within bounds
                @test violations == 0
            end
        end
    end