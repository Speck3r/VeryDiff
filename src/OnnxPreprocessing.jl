using MLStyle
import VNNLib

abstract type Layer end

struct Network
    layers::Vector{Layer}
end

struct Dense <: Layer
    W::Matrix{Float64}
    b::Vector{Float64}
end

struct ReLU <: Layer end

function (N::Network)(x :: Vector{Float64})
    for L in N.layers
        x = L(x)
    end
    return x
end

function (L::Dense)(x :: Vector{Float64})
    return L.W * x .+ L.b
end

function (L::ReLU)(x :: Vector{Float64})
    return max.(x,0.0)
end

function preprocess_onnx_model(net :: OnnxNet)
    # dictionary mapping output names (not node names!) to their values
    layers = Vector{Layer}()

    node_names = collect(keys(net.nodes))

    if length(net.start_nodes) != 1
        error("VeryDiff currently only supports networks with a single start node.\n
        Please ensure that your ONNX model has a single output node.\n
        Found: $(length(net.start_nodes)) start nodes: $(net.start_nodes)")
    end

    next_node = net.start_nodes[1]
    
    while !isnothing(next_node)
        node = net.nodes[next_node]
        next_node = nothing

        # Process Node
        (@match node begin
            _::VNNLib.OnnxParser.ONNXLinear => begin
                W = node.dense.weight
                b = node.dense.bias
                # TODO(steuber): Always omit?
                # if node.transpose
                #     W = W'
                # end
                push!(layers, Dense(W, b))
            end
            _::VNNLib.OnnxParser.ONNXAddConst => begin
                # Dense layer with identity matrix and constant bias
                W = I(size(node.c, 1))
                b = node.c
                push!(layers, Dense(W, b))
            end
            _::VNNLib.OnnxParser.ONNXSubConst => begin
                # Dense layer with identity matrix and constant bias
                W = I(size(node.c, 1))
                b = node.c
                if node.left
                    b = -b
                else
                    W = -W
                end
                push!(layers, Dense(W, b))
            end
            _::VNNLib.OnnxParser.ONNXRelu => begin
                push!(layers, ReLU())
            end
            _::VNNLib.OnnxParser.ONNXSoftmax => begin
                @warn "Softmax layer detected, which is only implicitly supported by VeryDiff -> Skipping parsing!"
                @assert node.name in net.final_nodes "Softmax output must be a final node in the ONNX model"
            end
            _ => error("Unsupported node type: $(node)")
        end)
        @assert length(node.outputs) == 1 "Expected exactly one output for each node, found $(length(node.outputs)) outputs"
        if node.name ∉ net.final_nodes
            node_candidates = filter(n -> node.outputs[1] in n[2].inputs, net.nodes)
            if length(node_candidates) == 0
                error("Output node $(node.outputs[1]) is not connected to any other nodes in the ONNX model")
            elseif length(node_candidates) > 1
                error("Output node $(node.outputs[1]) is connected to multiple nodes in the ONNX model: $(node_candidates)")
            end
            # this node is not a final node, so we can continue processing
            next_node = first(node_candidates)[1]
        end
    end

    return Network(layers)
end