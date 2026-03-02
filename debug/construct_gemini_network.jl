
using VNNLib
const OXP = VNNLib.OnnxParser

net1_path = joinpath(@__DIR__, "..", "test", "examples", "networks", "mnist_256x4_1e4.onnx")
net2_path = joinpath(@__DIR__, "..", "test", "examples", "networks", "mnist_256x4_2e5.onnx")
model1 = load_onnx_model(net1_path)
model2 = load_onnx_model(net2_path)

abstract type ONNXPoly{S,N<:Number} <: OXP.Node{S} end 

struct ONNXMonomialPoly{S,N<:Number} <: ONNXPoly{S,N}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S 
    coeffs::Array{N}
end

OXP.onnx_node_to_flux_layer(node::ONNXMonomialPoly) = x -> begin
    n_neurons, n_coeffs = size(node.coeffs)
    degree = n_coeffs - 1
    vec(sum(node.coeffs .* x .^ collect(0:degree)', dims=2))
end

struct ONNXChebyshevPoly{S,N<:Number,VN<:AbstractVector{N}} <: ONNXPoly{S,N}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S 
    coeffs::Array{N}
    l::VN 
    u::VN 
end

OXP.onnx_node_to_flux_layer(node::ONNXChebyshevPoly) = x -> begin
    
end

struct LayeredModel{S}
    layers :: Vector{OXP.Node{S}}
end


function to_layered_model(onnx_model::OnnxNet{S,N1,N2}) where {S,N1,N2}
    layers = Vector{OXP.Node{S}}()

    @assert length(onnx_model.start_nodes) == 1 "Only single input models are supported! Got $(onnx_model.start_nodes)"
    node_name = onnx_model.start_nodes[1]

    while haskey(onnx_model.node_nexts, node_name) && length(onnx_model.node_nexts[node_name]) > 0
        node = onnx_model.nodes[node_name]
        push!(layers, node)

        next_nodes = onnx_model.node_nexts[node_name]
        @assert length(next_nodes) == 1 "Only sequential models are supported! Got $(next_nodes)"
        node_name = onnx_model.node_nexts[node_name][1]
    end

    return LayeredModel(layers)
end


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

function to_gemini_network(model1::OnnxNet{S,N1,N2}, model2::OnnxNet{S,N1,N2}) where {S,N1,N2}
    lmodel1 = to_layered_model(model1)
    lmodel2 = to_layered_model(model2)

    @assert length(lmodel1.layers) == length(lmodel2.layers) "Both networks should have the same number of layers!"
    layersΔ = Vector{OXP.Node{S}}()
    for (l1, l2) in zip(lmodel1.layers, lmodel2.layers)
        lΔ = get_difference_layer(l1, l2)
        push!(layersΔ, lΔ)
    end

    return (lmodel1, lmodel2, LayeredModel(layersΔ))
end

