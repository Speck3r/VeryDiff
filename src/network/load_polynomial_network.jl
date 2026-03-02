
"""
Reads coefficients of polynomials for a given onnx model and polynomial coefficients in a set of csv files.

args:
    nn - the ReLU network whose ReLU's should be replaced by the polynomials with the specified coefficients
    poly_dir - directory containing coeffs_i.csv files for each layer i of the original network 

returns:
    list of matrices of polynomial coefficients for each layer
"""
function read_poly_coeffs(nn::LayeredModel, poly_dir)
    activation_layers = sum(isactivation.(nn.layers))

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
Loads bounds used to fit polynomials for approximation of the activation function.

args:
    nn - network, whose activations should be replaced by polynomials 
    poly_dir - directory containing range_i.csv files for each layer i of the original network 

returns:
    bounds - vector of n×2 matrices where bounds[i][1] are the lower bounds of the i-th layer 
             and bounds[i][2] are the upper bounds of the i-th layer.
             Bounds for non-activation layers are 0x0 matrices
"""
function load_approximation_bounds(nn::LayeredModel, poly_dir)
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



"""
Extracts the approximation domain for the polynomial activations of the given network.

Polynomial approximations quickly diverge to ±∞ outside of the approximation domain, therefore
we need to know the bounds of the approximation domain (pre-activation values for the original activation function) for each layer.

returns:
    list of n×2 matrices where bounds[i][1], bounds[i][2] are the lower and upper bounds of the approximation domain
     of the i-th layer or nothing if the layer is not a polynomial activation layer.
"""
function extract_approximation_domain(net::LayeredModel)
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
    model = load_onnx_model(model_file)
    nn = to_layered_model(model)
    coeffs = read_poly_coeffs(nn, poly_dir)

    layers_new = Vector{OXP.Node}()
    cnt = 1
    for L in nn.layers
        if isactivation(L)
            player = ONNXMonomialPoly(coeffs[cnt])
            cnt += 1

            push!(layers_new, player)
        else
            # TODO: does this cause problems because it's no copy?
            push!(layers_new, L)
        end
    end

    return LayeredModel(layers_new)
end