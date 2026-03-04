
"""
For each layer in the network, compute the interval bounds after that layer was applied.

args:
    net - Network to get bounds from 
    input_set - input set for which to get bounds 

returns:
    bounds - list of (n_neurons x 2)-array for each layer holding lower and upper bounds for each neuron 
             after that layer was applied
"""
function get_zono_bounds(net::LayeredModel, input_set::Zonotope)
    bounds = []
    
    prop_state = PropState(true)
    for i in 1:length(net.layers)
        layers = net.layers[1:i]
        net_partial = LayeredModel(layers)

        ẑ = net_partial(input_set, prop_state)
        bounds_layer = zono_bounds(ẑ)
        push!(bounds, bounds_layer)
    end

    return bounds
end


function approximate_polynomial(L::OXP.ONNXLinear, bounds, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
    # nothing to do here
    # no error for linear layer
    return L, 0.
end

function approximate_polynomial(L::OXP.ONNXRelu, bounds, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
    lower = @view bounds[:,1]
    upper = @view bounds[:,2]

    if max_polys_per_layer == 1
        lower = [minimum(lower)]
        upper = [maximum(upper)]
    end

    res = VeryDiff.approx_relu_poly.(lower, upper, degree, max_iter=max_iter, cheby=cheby)
    ps = hcat(getindex.(res, 1)...)'  # TODO: is there a better way to do vec of vec to matrix?
    ϵs = getindex.(res, 2)  # don't really need them, just for debugging 

    max_idx = argmax(ϵs)
    verbosity > 0 && println("max error = ", ϵs[max_idx], " at idx ", max_idx, " with bounds ", lower[max_idx], " ", upper[max_idx])

    #verbosity > 0 && println("max error = ", maximum(ϵs))

    if max_polys_per_layer == 1
        # repeat the single polynomial for all neurons
        # (needed in current implementation of ChebyshevPoly for correct evaluation)
        ps = repeat(ps[1:1, :], size(bounds, 1), 1)
        lower = repeat(lower[1:1], size(bounds, 1))
        upper = repeat(upper[1:1], size(bounds, 1))
    end

    # We want the networks to be isomorphic and want to be able to recognize that purely from the node names.
    # Therefore, we reuse the same node names.
    layer = cheby ? ONNXChebyshevPoly(L.inputs, L.outputs, L.name, Matrix(ps), lower, upper) : ONNXMonomialPoly(L.inputs, L.outputs, L.name, Matrix(ps))
    return layer, ϵs
end


function approximate_polynomial(L::OXP.ONNXGelu, bounds, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
    @assert cheby "Currently, we only allow approximation in Chebyshev coefficients for GeLU layers!"
    
    lower = @view bounds[:,1]
    upper = @view bounds[:,2]

    if max_polys_per_layer == 1
        lower = [minimum(lower)]
        upper = [maximum(upper)]
    end

    if max_iter == 0
        gelu = x -> 0.5 * x * (1 + erf(x / sqrt(2)))
        p = chebyshev_coefficients.(gelu, lower, upper, degree)
        ps = hcat(p...)'
        ϵs = -1  # TODO: can we get the real error?

        # we don't know the error yet (could use an overapproximation/just a sampled approximation of the error?)
        verbosity > 0 && println("GeLU chebyshev approximation")
    else
        res = approx_gelu_poly.(lower, upper, degree, max_iter=max_iter, cheby=cheby)
        ps = hcat(getindex.(res, 1)...)'  # TODO: is there a better way to do vec of vec to matrix?
        ϵs = getindex.(res, 2)  # don't really need them, just for debugging 

        max_idx = argmax(ϵs)
        verbosity > 0 && println("max error = ", ϵs[max_idx], " at idx ", max_idx, " with bounds ", lower[max_idx], " ", upper[max_idx])
    end

    if max_polys_per_layer == 1
        # repeat the single polynomial for all neurons
        # (needed in current implementation of ChebyshevPoly for correct evaluation)
        ps = repeat(ps[1:1, :], size(bounds, 1), 1)
        lower = repeat(lower[1:1], size(bounds, 1))
        upper = repeat(upper[1:1], size(bounds, 1))
    end

    # We want the networks to be isomorphic and want to be able to recognize that purely from the node names.
    # Therefore, we reuse the same node names.
    layer = cheby ? ONNXChebyshevPoly(L.inputs, L.outputs, L.name, Matrix(ps), lower, upper) : ONNXMonomialPoly(L.inputs, L.outputs, L.name, Matrix(ps))
    return layer, ϵs
end


function approximate_polynomial(L::ONNXPoly, bounds, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
    @warn "skipping already polynomial layer (but degree might differ!/Cheby vs. Monomial Form might not match!)"
    # no error for already polynomial layer
    return L, 0.
end

function approximate_polynomial(L::OXP.ONNXFlatten, bounds, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
    # no error for flattening layer
    return L, 0. 
end


function approximate_polynomial(net::LayeredModel, bounds::AbstractVector, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
    # attention: bounds are bounds AFTER the layer
    # for ReLU layers, we need the bound after the linear layer before that, the linear layers don't need any bounds
    bounds = [[[]]; bounds[1:end-1]]
    layers = map(x -> approximate_polynomial(x[1], x[2], degree, cheby=cheby, verbosity=verbosity, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer)[1],
                 zip(net.layers, bounds))
    return LayeredModel(layers)
end


"""
Iteratively approximate each layer in the network by a polynomial of a given degree.

The input ranges for the approximation are verified bounds computed by zonotope propagation.

args:
    net - Network to approximate 
    input_set - input set for which to get bounds 
    degree - degree of polynomial approximation for ReLU layers

kwargs:
    verbosity - verbosity level (0: silent, 1: print first 5 lower and upper bounds)
    cheby - whether to use Chebyshev basis (true) or Monomial basis (false) for polynomial approximation
    max_iter - maximum number of iterations for Remez algorithm
    max_polys_per_layer - maximum number of different polynomials to use per layer
"""
function approximate_polynomial_iterative(net::LayeredModel{S}, input_set, degree; verbosity=0, cheby=true, max_iter=20, max_polys_per_layer=Inf) where S
    @assert (max_polys_per_layer == Inf) || (max_polys_per_layer == 1) "only max_polys_per_layer=1 (one polynomial for all neurons) or Inf (one polynomial for each neuron) supported currently"
    prop_state = PropState(true)
    layers_poly = Vector{OXP.Node{S}}()
    ẑ = input_set
    for i in 1:length(net.layers)
        bounds_layer = zono_bounds(ẑ)

        verbosity > 0 && println("--- layer $i ---")
        verbosity > 0 && println("lower = ", bounds_layer[:,1][1:min(size(bounds_layer, 1), 5)])
        verbosity > 0 && println("upper = ", bounds_layer[:,2][1:min(size(bounds_layer, 1), 5)])
        !all(isfinite.(bounds_layer)) && println("lb non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,1])])
        !all(isfinite.(bounds_layer)) && println("ub non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,2])])

        layer = net.layers[i]
        layer_poly, ϵs = approximate_polynomial(layer, bounds_layer, degree, cheby=cheby, verbosity=verbosity, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer)
        push!(layers_poly, layer_poly)

        ẑ = layer_poly(ẑ, prop_state)
    end

    return LayeredModel(layers_poly)
end


function approximate_polynomial_iterative_sampling(net::LayeredModel{S}, X_in::AbstractVector, degree; verbosity=0, cheby=true, max_iter=20, max_polys_per_layer=Inf) where S
    @assert (max_polys_per_layer == Inf) || (max_polys_per_layer == 1) "only max_polys_per_layer=1 (one polynomial for all neurons) or Inf (one polynomial for each neuron) supported currently"
    layers_poly = Vector{OXP.Node{S}}()
    ys_layer = X_in 
    Y_layer = hcat(ys_layer...)'
    for i in 1:length(net.layers)
        lb_layer = minimum(Y_layer, dims=1)'
        ub_layer = maximum(Y_layer, dims=1)' 
        bounds_layer = hcat(lb_layer, ub_layer)

        verbosity > 0 && println("--- layer $i ---")
        verbosity > 0 && println("lower = ", bounds_layer[:,1][1:min(size(bounds_layer, 1), 5)])
        verbosity > 0 && println("upper = ", bounds_layer[:,2][1:min(size(bounds_layer, 1), 5)])
        !all(isfinite.(bounds_layer)) && println("lb non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,1])])
        !all(isfinite.(bounds_layer)) && println("ub non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,2])])

        layer = net.layers[i]
        layer_poly, ϵs = approximate_polynomial(layer, bounds_layer, degree, cheby=cheby, verbosity=verbosity, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer)
        push!(layers_poly, layer_poly)

        # TODO: can flux evaluate this in batch mode?
        ys_layer = [OXP.onnx_node_to_flux_layer(layer_poly)(y) for y in ys_layer]
        Y_layer = hcat(ys_layer...)'
    end

    return LayeredModel(layers_poly)   
end


"""
Iteratively approximate each layer in the network by a polynomial of a given degree with pre-activation ranges computed via alpha-beta-CROWN.

The pre-activation ranges are computed using additional error inputs added to the network after every activation layer.

args:
    onnx_path - Path to ONNX network to approximate
    degree - degree of polynomial approximation for ReLU layers

kwargs:
    verbosity - verbosity level (0: silent, 1: print first 5 lower and upper bounds)
    cheby - whether to use Chebyshev basis (true) or Monomial basis (false) for polynomial approximation
    max_iter - maximum number of iterations for Remez algorithm
    max_polys_per_layer - maximum number of different polynomials to use per layer
    tight_gelu - use tight initialization of gelu relaxation
"""
function approximate_polynomial_abcrown(onnx_path, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf, tight_gelu=true)
    ERROR_NODES_SCRIPT = pyimport("insert_error_nodes")
    BOUNDS_SCRIPT      = pyimport("get_bounds")

    # load model in julia
    model = load_onnx_model(onnx_path)
    lmodel = to_layered_model(model)

    # extend network by error inputs
    error_net_path = "error_net.onnx"
    ERROR_NODES_SCRIPT.insert_error_nodes(onnx_path, error_net_path)
    # run(`$(ABCROWN_PYTHONPATH) $(ERROR_NODES_SCRIPT) $(onnx_path) $(error_net_path)`)

    input_bounds_file = "input_bounds.h5"
    activations_considered = []
    error_magnitudes = []
    layer_bounds = []
    layers_poly = Vector{OXP.Node}()

    for (i, l) in enumerate(lmodel.layers)
        if (l isa OXP.ONNXRelu) || (l isa OXP.ONNXGelu)
            # prepare input bounds
            h5open(input_bounds_file, "w") do file 
                for (k, v) in model.input_shapes
                    # TODO: make this more general than just [0,1] input bounds
                    lb = zeros(v)
                    ub = ones(v)
                    # hdf5 already converts from WHCN to NCHW
                    file[k] = cat(lb, ub, dims=ndims(lb))
                end

                for (l, ϵs) in zip(activations_considered, error_magnitudes)
                    error_name = "eps_" * l.name
                    file[error_name] = reshape(ϵs, :, 1)  # append a batch dimension
                end
            end

            output_name = l.inputs[1]
            output_bounds = "out_bounds.h5"
            BOUNDS_SCRIPT.compute_pre_activation_bounds(error_net_path, input_bounds_file, output_name; outfile=output_bounds, method="alpha-crown", tight_gelu=tight_gelu)
            # run(`$ABCROWN_PYTHONPATH $(BOUNDS_SCRIPT) $(error_net_path) $(input_bounds_file) $(output_name) --output_file $(output_bounds)`)

            bounds = h5open(output_bounds, "r") do file 
                read(file[output_name])
            end

            # ϵs = compute_approximation_errors(bounds, ϵ=0.05)
            layer_poly, ϵs = VeryDiff.approximate_polynomial(l, bounds, degree, cheby=cheby, verbosity=verbosity, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer)

            push!(activations_considered, l)
            push!(error_magnitudes, ϵs)
            push!(layer_bounds, bounds)

            verbosity > 0 && println("--- layer $i ---")
            verbosity > 0 && println("lower = ", bounds[:,1][1:min(size(bounds, 1), 5)])
            verbosity > 0 && println("upper = ", bounds[:,2][1:min(size(bounds, 1), 5)])
            !all(isfinite.(bounds)) && println("lb non-finite: ", (1:size(bounds,1))[.~isfinite.(bounds[:,1])])
            !all(isfinite.(bounds)) && println("ub non-finite: ", (1:size(bounds,1))[.~isfinite.(bounds[:,2])])
        else
            layer_poly = l
        end
        push!(layers_poly, layer_poly)
    end

    # need to do [l for l in layers_poly] to convert from OXP.Node without information about identifier type to OXP.Node{S}
    return LayeredModel([l for l in layers_poly])
end
