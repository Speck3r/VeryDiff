using VeryDiff
using VNNLib
import VNNLib.NNLoader: load_network
using Distributions
using Random
using LinearAlgebra

using Enzyme

#Enzyme.API.printactivity!(true)
#Enzyme.API.printall!(true)

function prepare_weight_mutation(NN)
    generator = Uniform(9e-7, 2e-6)
    Random.seed!(1234)
    j = 0
    inputs = Matrix{Float64}[]
    for (i, layer) in enumerate(NN.layers)
        if layer isa VNNLib.Dense
            j += 1
            push!(inputs, ones(size(layer.W)))
        end
    end
    for i in axes(inputs,1)
        inputs[i] .= 1.0 .- rand(generator, size(inputs[i]))
    end
    return inputs, j
end


function mutate_weights(NN, inputs)
    layers = []
    j = 1
    for layer in NN.layers
        if layer isa VNNLib.Dense
            println(size(inputs[j]))
            W_new = layer.W .* inputs[j]
            j += 1
            b_new = layer.b
            push!(layers, VNNLib.Dense(W_new, b_new))
        else
            push!(layers, layer)
        end
    end
    return VeryDiff.Network(layers)
end

function compile_autodiff()

    VeryDiff.NEW_HEURISTIC = false

    sysimage_dir = @__DIR__

    NN1 = load_network("$sysimage_dir/../../test/examples/nets/acc-almost-safe.onnx")

    NN1 = Network(NN1.layers[1:2]) # Small network for testing

    input, j = prepare_weight_mutation(NN1)
    println("Number of DiffRealDense layers: $j")
    NN2 = mutate_weights(NN1, input)

    N = GeminiNetwork(NN1,NN2)

    spec, n_inputs, _ = get_ast("$sysimage_dir/../../test/examples/specs/acc.vnnlib")

    ((bounds, _, _, num), _) = iterate(spec)

    low = @view bounds[1:n_inputs,1]
    high = @view bounds[1:n_inputs,2]
    mid = (high.+low) ./ 2
    distance = mid .- low
    non_zero_indices = findall((!).(iszero.(distance)))
    distance = distance[non_zero_indices]
    ∂Z_original = Zonotope(Matrix{Float64}(0.0I,n_inputs,size(non_zero_indices,1)),zeros(Float64,n_inputs),nothing)
    Z1 = Zonotope(Matrix{Float64}(I, n_inputs, n_inputs)[:,non_zero_indices] .* distance, mid, nothing)
    Z2 = deepcopy(Z1)
    diff_zono = VeryDiff.DiffZonotope(Z1, Z2, ∂Z_original, 0, 0, 0)

    P = VeryDiff.PropState(true)

    VeryDiff.forward_diff_diff_bound_loss(N, diff_zono, P)

    diff_forward_result = VeryDiff.forward_diff_diff_bound_loss(N, diff_zono, P)
    println("diff_forward loss: ", diff_forward_result)

    N_copy = Enzyme.make_zero(N)
    Z_copy = Enzyme.make_zero(diff_zono)

    P = VeryDiff.PropState(true)
    @info "Compiling autodiff for VeryDiff.diff_bound_loss..."
    
    gs = @time Enzyme.autodiff(
        set_runtime_activity(ReverseWithPrimal),
        VeryDiff.forward_diff_diff_bound_loss,
        Active,
        Duplicated(N, N_copy),
        Duplicated(diff_zono, Z_copy),
        Const(P)
    )

    # println("Gradients: ", gs)
end


fetch(schedule(Task(() ->
compile_autodiff(),
1024 * 1024 * 1024)))