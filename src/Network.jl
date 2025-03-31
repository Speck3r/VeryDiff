

isactivation(L::ReLU) = true
isactivation(L::Dense) = false

abstract type Poly{N<:Number} <: VNNLib.Layer end

# since polynomial networks are not defined in VNNLib.jl, we define their behaviour for concrete inputs here 

struct MonomialPoly{N<:Number} <: Poly{N}
    coeffs::Array{N}
end


function (L::MonomialPoly)(x::Vector{N}) where {N<:Number}
    # TODO: this forces all polynomials in a layer to have the same degree (or zero coeffs) (do we want that?)
    n_neurons, n_coeffs = size(L.coeffs)
    degree = n_coeffs - 1
    vec(sum(L.coeffs .* x .^ collect(0:degree)', dims=2))
end


"""
Polynomial activation layer with coefficients of polynomial stored w.r.t. Chebyshev basis.

The domain of the Chebyshev polynomials is NOT scaled! (i.e. coefficients are stored w.r.t x ∈ [-1, 1],
however they can also be **evaluated** for x ∉ [-1, 1])

attrs:
    coeffs - array of coefficients, s.t. pᵢ(x) = c₀T₀(x) + c₁T₁(x) + ...
"""
struct ChebyshevPoly{N<:Number,VN<:AbstractVector{N}} <: Poly{N} 
    coeffs::Array{N}
    l::VN
    u::VN
end


function ChebyshevPoly(coeffs::Array{N}) where N<:Number
    m, n = size(coeffs)
    l = .-ones(N, m)
    u = ones(N, m)

    ChebyshevPoly(coeffs, l, u)
end


function (L::ChebyshevPoly)(x::Vector{N}) where {N<:Number}
    clenshaw_chebyshev.(eachrow(L.coeffs), x, L.l, L.u)
end


isactivation(L::Poly) = true


struct DiffLayer{L1<:VNNLib.Layer,L2<:VNNLib.Layer} <: VNNLib.Layer
    layer1::L1
    layer2::L2
end


function (L::DiffLayer)(x::Vector{N}) where N<:Number
    y₁ = L.layer1(x)
    y₂ = L.layer2(x)
    y₁ .- y₂
end


struct PolyReLU{N} <: VNNLib.Layer where {N<:Number}
    coeffs::Array{N}
end


function (L::PolyReLU)(x::Vector{N}) where {N<:Number}
    y₁ = Poly(L.coeffs)(x)
    y₂ = max.(zero(N), x)
    y₁ .- y₂
end


"""
Reads coefficients of polynomials for a given onnx model and polynomial coefficients in a set of csv files.

args:
    nn - the ReLU network whose ReLU's should be replaced by the polynomials with the specified coefficients
    poly_dir - directory containing coeffs_i.csv files for each layer i of the original network 

returns:
    list of matrices of polynomial coefficients for each layer
"""
function read_poly_coeffs(nn, poly_dir)
    #nn = VNNLib.load_network(model_file)
    activation_layers = sum(typeof.(nn.layers) .== VNNLib.ReLU)

    coeffs = []
    for layer in 0:activation_layers-1
        f = CSV.File(string(poly_dir, "/coeffs_", layer, ".csv"), header=false)
        n_neurons = length(f)
        n_coeffs = n_neurons > 0 ? length(f[1]) : 0

        coeffs_layer = zeros(n_neurons, n_coeffs)
        for (i, line) in enumerate(f)
            for j in 1:length(line)
                coeffs_layer[i, j] = line[j]
            end
        end

        push!(coeffs, coeffs_layer)
    end

    return coeffs
end


"""
Loads bounds used to fit polynomials for approximation of the ReLU function.

args:
    nn - ReLU network, whose activations should be replaced by polynomials 
    poly_dir - directory containing range_i.csv files for each layer i of the original network 

returns:
    bounds - vector of n×2 matrices where bounds[i][1] are the lower bounds of the i-th layer 
             and bounds[i][2] are the upper bounds of the i-th layer.
             Bounds for non-activation layers are 0x0 matrices
"""
function load_approximation_bounds(nn, poly_dir)
    n_layers = length(nn.layers)
    act_idxs = [i for (L, i) in zip(nn.layers, 1:n_layers) if isactivation(L)]
    activation_layers = length(act_idxs)

    bounds = [Matrix{Float64}(undef, 0, 0) for i in 1:n_layers]
    for (act_layer, idx) in zip(0:activation_layers-1, act_idxs)
        f = CSV.File(string(poly_dir, "/range_", act_layer, ".csv"), header=false)
        n_neurons = length(f)

        bounds_layer = zeros(n_neurons, 2)
        for (i, line) in enumerate(f)
            bounds_layer[i,1] = line[1]
            bounds_layer[i,2] = line[2]
        end

        bounds[idx] = bounds_layer
    end

    return bounds   
end


function extract_approximation_domain(L::ReLU)
    return nothing    
end

function extract_approximation_domain(L::ChebyshevPoly)
    return L.l, L.u
end

function extract_approximation_domain(L::MonomialPoly)
    return nothing
end

function extract_approximation_domain(L::Dense)
    return nothing
end


"""
Extracts the approximation domain for the polynomial activations of the given network.

Polynomial approximations quickly diverge to ±∞ outside of the approximation domain, therefore
we need to know the bounds of the approximation domain (pre-activation values for the original activation function) for each layer.

returns:
    list of n×2 matrices where bounds[i][1], bounds[i][2] are the lower and upper bounds of the approximation domain
     of the i-th layer or nothing if the layer is not a polynomial activation layer.
"""
function extract_approximation_domain(net::Network)
    res = extract_approximation_domain.(net.layers)
    [isnothing(r) ? nothing : [r[1] r[2]] for r in res]
end



"""
Loads network whose activation functions have been replaced by the polynomials with 
coefficients specified in the given directory.

args:
    model_file - .onnx file containing the ReLU network 
    poly_dir - directory containing coeffs_i.csv files for each layer i of the original network 

returns:
    VNNLib.Network with polynomial activations
"""
function load_polynomial_nn(model_file, poly_dir)
    nn = VNNLib.load_network(model_file)
    coeffs = read_poly_coeffs(nn, poly_dir)

    layers_new = Vector{VNNLib.Layer}()
    cnt = 1
    for L in nn.layers
        if typeof(L) == VNNLib.ReLU
            player = MonomialPoly(coeffs[cnt])
            cnt += 1

            push!(layers_new, player)
        else
            # TODO: does this cause problems because it's no copy?
            push!(layers_new, L)
        end
    end

    return VNNLib.Network(layers_new)
end


"""
Computes all intermediate preactivation and activation values for input x in the network.

args:
    net - network to execute 
    x - input vector 

returns:
    xs - vector of length length(net.layers) with outputs of each layer
"""
function intermediate_activations(net::Network, x::AbstractVector)
    accfun = (L, xs) -> begin
        x = L(xs[end])
        push!(xs, x)
    end

    xs = foldl((xs, L) -> accfun(L, xs),net.layers,init=[x])
    # we already know the input values, so throw them away
    return xs[2:end]
end


function parse_network(n::Network)
    return n
end