

# since polynomial networks are not defined in VNNLib.jl, we define their behaviour for concrete inputs here 

struct Poly{N} <: VNNLib.Layer where {N<:Number}
    coeffs::Array{N}
end


function (L::Poly)(x::Vector{N}) where {N<:Number}
    # TODO: this forces all polynomials in a layer to have the same degree (or zero coeffs) (do we want that?)
    n_neurons, n_coeffs = size(L.coeffs)
    degree = n_coeffs - 1
    vec(sum(L.coeffs .* x .^ collect(0:degree)', dims=2))
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


function load_approximation_bounds(nn, poly_dir)
    activation_layers = sum(typeof.(nn.layers) .== VNNLib.ReLU)

    lbs = Vector{Vector{Float64}}(undef, activation_layers)
    ubs = Vector{Vector{Float64}}(undef, activation_layers)
    for layer in 0:activation_layers-1
        f = CSV.File(string(poly_dir, "/range_", layer, ".csv"), header=false)
        n_neurons = length(f)

        lbs_layer = fill(-Inf, n_neurons)
        ubs_layer = fill(Inf, n_neurons)
        for (i, line) in enumerate(f)
            lbs_layer[i] = line[1]
            ubs_layer[i] = line[2]
        end

        lbs[layer+1] = lbs_layer
        ubs[layer+1] = ubs_layer
    end

    return lbs, ubs    
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
    coeffs = read_poly_coeffs(model_file, poly_dir)

    layers_new = Vector{VNNLib.Layer}()
    cnt = 1
    for L in nn.layers
        if typeof(L) == VNNLib.ReLU
            player = Poly(coeffs[cnt])
            cnt += 1

            push!(layers_new, player)
        else
            # TODO: does this cause problems because it's no copy?
            push!(layers_new, L)
        end
    end

    return VNNLib.Network(layers_new)
end


function parse_network(n::Network)
    return n
end