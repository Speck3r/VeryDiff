
# For now, VeryDiff is only applicable to layered architectures and not full computational graphs.
# To this end, we have our own type LayeredModel 

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


"""
Computes all intermediate preactivation and activation values for input x in the network.

args:
    net - network to execute 
    x - input vector 

returns:
    xs - vector of length length(net.layers) with outputs of each layer
"""
function intermediate_activations(net::LayeredModel, x::AbstractVector)
    accfun = (L, xs) -> begin
        f = onnx_node_to_flux_layer(L)
        x = f(xs[end])
        push!(xs, x)
    end

    xs = foldl((xs, L) -> accfun(L, xs),net.layers,init=[x])
    # we already know the input values, so throw them away
    return xs[2:end]
end