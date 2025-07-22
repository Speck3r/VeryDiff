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

function preprocess_onnx_model(network :: VNNLibNetwork)

end