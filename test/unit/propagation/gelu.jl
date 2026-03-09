
using VNNLib
using VNNLib.OnnxParser: ONNXLinear, ONNXAddConst, ONNXRelu, ONNXGelu, Node
using VeryDiff.Definitions: VerificationTask
using VeryDiff: approximate_polynomial_iterative
using Random 
const OXP = VNNLib.OnnxParser
# include(joinpath(@__DIR__, "../test/unit/propagation/utils.jl"))

function fuzz_gelu_poly_diff(input_dim, num_layers, degree; num_samples=1000)
    leading_coeff_warning = VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[]
    VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false


    l, u = -ones(input_dim), ones(input_dim)

    input_samples = sample_points_in_hypercube(l, u, num_samples)

    layer_dims = [rand(20:50) for _ in 1:(num_layers-1)]
    push!(layer_dims, 10)  # output dimension is always 10

    N1, N2 = make_dense_pair(input_dim, layer_dims, activation=:gelu, identical=true)
    N1 = approximate_polynomial_iterative(N1, l, u, degree, verbosity=0)

    N_gemini = GeminiNetwork(N1, N2)

    for i in 1:num_layers
        N_cur = GeminiNetwork(N_gemini.inputs, deepcopy(N_gemini.diff_layers[1:i]))
        task = create_verification_task(l, u)

        P = VeryDiff.PropState(true)
        VeryDiff.prepare_prop_state!(P, task)
        VeryDiff.init_bounds_cache_approximation_domain!(N_cur, P)
        P = VeryDiff.propagate!(N_cur, P)
        Zout = P.zono_storage.zonotopes[end].zonotope

        bounds_z1 = zono_bounds(Zout.Z₁)
        bounds_z2 = zono_bounds(Zout.Z₂)
        bounds_z∂ = zono_bounds(Zout.∂Z)

        n1_violations = 0
        n2_violations = 0
        diff_violations = 0
        agnostic_n1_violations = 0
        agnostic_n2_violations = 0
        agnostic_diff_violations = 0
        for j in 1:num_samples
            x = input_samples[:, j]

            Zin = P.zono_storage.zonotopes[1].zonotope.Z₁
            @assert Zin.c .+ Zin.Gs[1]*x ≈ x atol=1e-8
            Zin2 = P.zono_storage.zonotopes[1].zonotope.Z₂
            @assert Zin2.c .+ Zin2.Gs[1]*x ≈ x atol=1e-8

            in_name = first(N1.input_shapes)[1]
            out_name = N_cur.diff_layers[end].layer1.outputs[1]
            y1_dict = OXP.compute_all_outputs(N1, Dict(in_name => x))
            y1 = y1_dict[out_name]
            y2_dict = OXP.compute_all_outputs(N2, Dict(in_name => x))
            y2 = y2_dict[out_name]

            Z1_range = sum(g->sum(abs, g, dims=2),Zout.Z₁.Gs[2:end];init=zeros(size(Zout.Z₁.c)))
            Z2_range = sum(g->sum(abs, g, dims=2),Zout.Z₂.Gs[2:end];init=zeros(size(Zout.Z₂.c)))
            input_component1 = Zout.Z₁.c .+ Zout.Z₁.Gs[1]*x
            input_component2 = Zout.Z₂.c .+ Zout.Z₂.Gs[1]*x

            diff_range = sum(g->sum(abs, g, dims=2),Zout.∂Z.Gs[2:end];init=zeros(size(Zout.∂Z.c)))
            if length(Zout.∂Z.Gs) >= 1
                input_component_diff = Zout.∂Z.c .+ Zout.∂Z.Gs[1]*x
            else
                input_component_diff = Zout.∂Z.c
            end

            mask_1_lb = (input_component1 .- Z1_range .<= y1 .+ 1e-8)
            mask_1_ub = (y1 .<= input_component1 .+ Z1_range .+ 1e-8)
            mask_2_lb = (input_component2 .- Z2_range .<= y2 .+ 1e-8)
            mask_2_ub = (y2 .<= input_component2 .+ Z2_range .+ 1e-8)
            mask_∂_lb = ((input_component_diff .- diff_range) .<= (y1 .- y2) .+ 1e-8) 
            mask_∂_ub = ((y1 .- y2) .<= (input_component_diff .+ diff_range) .+ 1e-8) 

            n1_violations += count(.~mask_1_lb)
            n1_violations += count(.~mask_1_ub)
            n2_violations += count(.~mask_2_lb)
            n2_violations += count(.~mask_2_ub)
            diff_violations += count(.~mask_∂_lb)
            diff_violations += count(.~mask_∂_ub)
            any(.~mask_1_lb) && @info "lower bound violated for Z₁"
            any(.~mask_1_ub) && @info "upper bound violated for Z₁"
            any(.~mask_2_lb) && @info "lower bound violated for Z₂"
            any(.~mask_2_ub) && @info "upper bound violated for Z₂"
            any(.~mask_∂_lb) && @info "lower bound violated for ∂Z"
            any(.~mask_∂_ub) && @info "upper bound violated for ∂Z"


            
            mask_n1_lb = bounds_z1[:,1] .<= y1 .+ 1e-8
            mask_n1_ub = y1 .<= bounds_z1[:,2] .+ 1e-8
            mask_n2_lb = bounds_z2[:,1] .<= y2 .+ 1e-8
            mask_n2_ub = y2 .<= bounds_z2[:,2] .+ 1e-8
            mask_diff_lb = bounds_z∂[:,1] .<= (y1 .- y2) .+ 1e-8
            mask_diff_ub = (y1 .- y2) .<= bounds_z∂[:,2] .+ 1e-8

            any(.~mask_n1_lb) && @info "agnostic lower bound violated for Z₁"
            any(.~mask_n1_ub) && @info "agnostic upper bound violated for Z₁"
            any(.~mask_n2_lb) && @info "agnostic lower bound violated for Z₂"
            any(.~mask_n2_ub) && @info "agnostic upper bound violated for Z₂"
            any(.~mask_diff_lb) && @info "agnostic lower bound violated for ∂Z"
            any(.~mask_diff_ub) && @info "agnostic upper bound violated for ∂Z"

            agnostic_n1_violations += count(.~mask_n1_lb)
            agnostic_n1_violations += count(.~mask_n1_ub)
            agnostic_n2_violations += count(.~mask_n2_lb)
            agnostic_n2_violations += count(.~mask_n2_ub)
            agnostic_diff_violations += count(.~mask_diff_lb)
            agnostic_diff_violations += count(.~mask_diff_ub)
        end

        violations = n1_violations + n2_violations + diff_violations + agnostic_n1_violations + agnostic_n2_violations + agnostic_diff_violations
        @info "Found $violations bound violations!"
        @test violations == 0
    end

    # restore prior settings
    VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = leading_coeff_warning
end


function fuzz_gelu_poly_diff(;rounds=10)
    if "VERYDIFF_TEST_SEED" in keys(ENV)
        test_seed = parse(Int, ENV["VERYDIFF_TEST_SEED"])
        @info "Using VERYDIFF_TEST_SEED: $(test_seed)"
    else
        test_seed = rand(1:999999)
    end
    Random.seed!(test_seed)
    @info "Test seed: $(test_seed)"

    for i in 1:rounds
        @testset "Gelu-Poly: Sampled Points within Output bounds $i" begin 
            input_dim = rand(5:25)
            num_layers = rand(1:15)
            degree = rand(10:40)
            @info "Fuzzing with input_dim = $(input_dim), num_layers = $(num_layers), degree = $degree"
            fuzz_gelu_poly_diff(input_dim, num_layers, degree)
        end
    end     
end

fuzz_gelu_poly_diff(rounds=10)