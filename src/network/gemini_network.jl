
# We only allow difference computation for isomorphic networks.
# To avoid having to do graph isomorphism tests, we require the networks' nodes 
# to have the same names.
# In this way, we can easily verify the isomorphism.


function check_isomorphic_names(node1::OXP.Node, node2::OXP.Node)
    @assert node1.inputs == node2.inputs "Node inputs have different names! Got $(node1.inputs) vs $(node2.inputs)"
    @assert node1.outputs == node2.outputs "Node outputs have different names! Got $(node1.outputs) vs $(node2.outputs)"
    @assert node1.name == node2.name "Nodes have different names! Got $(node1.name) vs $(node2.name)"
end

function get_difference_layer(node1::OXP.Node{S}, node2::OXP.Node{S}) where S 
    check_isomorphic_names(node1, node2)
    @assert false "Unsupported combination of node types: $(typeof(node1)) and $(typeof(node2))!"
end

function get_difference_layer(node1::OXP.ONNXFlatten{S}, node2::OXP.ONNXFlatten{S}) where S 
    check_isomorphic_names(node1, node2)
    @assert node1.axis == node2.axis "Flatten layers have to flatten same axis! Got $(node1.axis) vs $(node2.axis)"
    OXP.ONNXFlatten(deepcopy(node1.inputs), deepcopy(node1.outputs), string(node1.name, "_Δ"), node1.axis)
end

function get_difference_layer(node1::OXP.ONNXLinear{S}, node2::OXP.ONNXLinear{S}) where S 
    check_isomorphic_names(node1, node2)
    @assert node1.transpose == node2.transpose "One node is transposed while the other is not!"

    W1 = node1.dense.weight 
    b1 = node1.dense.bias
    W2 = node2.dense.weight 
    b2 = node2.dense.bias 

    @assert size(W1) == size(W2) "Mismatch in weight matrix size: $(size(W1)) vs $(size(W2))"
    @assert size(b1) == size(b2)

    OXP.ONNXLinear(node1.inputs, node1.outputs, string(node1.name, "_Δ"), W1 .- W2, b1 .- b2, transpose=node1.transpose,  double_precision=eltype(W1) == Float64)
end

function get_difference_layer(node1::OXP.ONNXRelu{S}, node2::OXP.ONNXRelu{S}) where {S}
    check_isomorphic_names(node1, node2)
    OXP.ONNXRelu(deepcopy(node1.inputs), deepcopy(node1.outputs), string(node1.name, "_Δ"))
end

function get_difference_layer(node1::ONNXPoly{S}, node2::OXP.ONNXRelu{S}) where {S}
    check_isomorphic_names(node1, node2)
    ONNXDiffNode(node1, node2)
end

function to_gemini_network(lmodel1::LayeredModel{S}, lmodel2::LayeredModel{S}) where S
    @assert length(lmodel1.layers) == length(lmodel2.layers) "Both networks should have the same number of layers!"
    layersΔ = Vector{OXP.Node{S}}()
    for (l1, l2) in zip(lmodel1.layers, lmodel2.layers)
        lΔ = get_difference_layer(l1, l2)
        push!(layersΔ, lΔ)
    end

    return (lmodel1, lmodel2, LayeredModel(layersΔ))
end

function to_gemini_network(model1::OnnxNet{S,N1,N2}, model2::OnnxNet{S,N1,N2}) where {S,N1,N2}
    lmodel1 = to_layered_model(model1)
    lmodel2 = to_layered_model(model2)

    to_gemini_network(lmodel1, lmodel2)
end


function cleanup_network(network1)
    valid_layers = []
    for i in 1:length(network1.layers)
        if network1.layers[i] isa OXP.ONNXLinear
            if all(isone.(diag(network1.layers[i].dense.weight))) && all([all(iszero.(diag(network1.layers[i].dense.weight, k))) && all(iszero.(diag(network1.layers[i].dense.weight, -k))) for k in 1:size(network1.layers[i].dense.weight,1)-1])
                continue
            end
        end
        push!(valid_layers, i)
    end
    print(valid_layers)
    @assert length(valid_layers) == length(network2.layers)
    return LayeredModel(network1.layers[valid_layers])
end

struct GeminiNetwork
    network1 :: LayeredModel
    network2 :: LayeredModel
    diff_network :: LayeredModel
    function GeminiNetwork(lmodel1::LayeredModel, lmodel2::LayeredModel)
        if length(lmodel1.layers) > length(lmodel2.layers)
            lmodel1 = cleanup_network(lmodel1)
        elseif length(lmodel2.layers) > length(lmodel1.layers)
            lmodel2 = cleanup_network(lmodel2)
        end
        @assert length(lmodel1.layers) == length(lmodel2.layers)
        nn1, nn2, nnΔ = to_gemini_network(lmodel1, lmodel2)
        return new(nn1, nn2, nnΔ)
    end
end

function GeminiNetwork(network1::OnnxNet, network2::OnnxNet)
    lmodel1 = to_layered_model(network1)
    lmodel2 = to_layered_model(network2)
    GeminiNetwork(lmodel1, lmodel2)
end