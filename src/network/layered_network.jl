
# For now, VeryDiff is only applicable to layered architectures and not full computational graphs.
# To this end, we have our own type LayeredModel 

struct LayeredModel{S}
    layers :: Vector{OXP.Node{S}}
end


function to_layered_model(onnx_model::OnnxNet{S,N1,N2}) where {S,N1,N2}
    layers = Vector{OXP.Node{S}}()

    @assert length(onnx_model.start_nodes) == 1 "Only single input models are supported! Got $(onnx_model.start_nodes)"
    node_name = onnx_model.start_nodes[1]

    while haskey(onnx_model.node_nexts, node_name)
        node = onnx_model.nodes[node_name]

        if node isa OXP.ONNXFlatten
            # TODO: worry about flatten when we have convolutional networks
            @warn "Skipping Flatten layer!"
        else
            push!(layers, node)
        end

        next_nodes = onnx_model.node_nexts[node_name]
        # 0 next nodes are also allowed for the output layer
        @assert length(next_nodes) <= 1 "Only sequential models are supported! Got $(next_nodes)"
        length(next_nodes) == 0 && break
        node_name = onnx_model.node_nexts[node_name][1]
    end

    return LayeredModel(layers)
end


function (model::LayeredModel)(x::AbstractArray)
    # is creating a new flux layer every time efficient?
    return foldl((x,L) -> OXP.onnx_node_to_flux_layer(L)(x), model.layers,init=x)
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
        f = OXP.onnx_node_to_flux_layer(L)
        x = f(xs[end])
        push!(xs, x)
    end

    xs = foldl((xs, L) -> accfun(L, xs),net.layers,init=[x])
    # we already know the input values, so throw them away
    return xs[2:end]
end