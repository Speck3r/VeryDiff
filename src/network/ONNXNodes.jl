
# This file contains additions for VNNLib.OnnxParser, s.t. we can have 
# nodes/layers that represent differences and polynomial activations.
#
# The file also contains methods that are defined on the specific nodes:
# - isactivation(node)
# - extract_approximation_domain(node)


abstract type ONNXPoly{S,N<:Number} <: OXP.Node{S} end 

struct ONNXMonomialPoly{S,N<:Number} <: ONNXPoly{S,N}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S 
    coeffs::Array{N}
end

onnx_node_to_flux_layer(node::ONNXMonomialPoly) = x -> begin
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

onnx_node_to_flux_layer(node::ONNXChebyshevPoly) = x -> begin
    clenshaw_chebyshev.(eachrow(node.coeffs), x, node.l, node.u)
end

struct ONNXDiffNode{S,N1<:OXP.Node{S},N2<:OXP.Node{S}} <: OXP.Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    node1::N1
    node2::N2
end

function ONNXDiffNode(node1::OXP.Node{S}, node2::OXP.Node{S}) where S 
    inputs = [string(i, "_Δ") for i in node1.inputs]
    outputs = [string(o, "_Δ") for o in node1.outputs]
    name = string(node1.name, "_Δ")
    ONNXDiffNode(inputs, outputs, name, node1, node2)
end

function onnx_node_to_flux_layer(node::ONNXDiffNode)
    flux_layer1 = onnx_node_to_flux_layer(node.node1)
    flux_layer2 = onnx_node_to_flux_layer(node.node2)

    x -> begin
        y₁ = flux_layer1(x)
        y₂ = flux_layer2(x)
        y₁ .- y₂
    end
end



isactivation(L::ONNXPoly) = true
isactivation(L::OXP.ONNXRelu) = true 
isactivation(L::OXP.ONNXLinear) = false
isactivation(L::OXP.ONNXFlatten) = false



function extract_approximation_domain(L::OXP.ONNXRelu)
    return nothing    
end

function extract_approximation_domain(L::ONNXChebyshevPoly)
    return L.l, L.u
end

function extract_approximation_domain(L::ONNXMonomialPoly)
    return nothing
end

function extract_approximation_domain(L::OXP.ONNXLinear)
    return nothing
end

