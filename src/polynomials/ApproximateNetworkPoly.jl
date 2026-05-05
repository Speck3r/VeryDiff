
# TODO: What to do about polynomial approximation via zonotope bounds?
#       Currently commented out, because difficult to run zonos on a single NN in new VeryDiff.


"""
For each layer in the network, compute the interval bounds after that layer was applied.

args:
    net - Network to get bounds from 
    input_set - input set for which to get bounds 

returns:
    bounds - list of (n_neurons x 2)-array for each layer holding lower and upper bounds for each neuron 
             after that layer was applied
"""
# function get_zono_bounds(net::LayeredModel, input_set::Zonotope)
#     bounds = []
    
#     prop_state = PropState(true)
#     for i in 1:length(net.layers)
#         layers = net.layers[1:i]
#         net_partial = LayeredModel(layers)

#         ẑ = net_partial(input_set, prop_state)
#         bounds_layer = zono_bounds(ẑ)
#         push!(bounds, bounds_layer)
#     end

#     return bounds
# end


function approximate_polynomial(L::OXP.ONNXLinear, bounds, degree; cheby=true, verbosity=0, selection=:contiguous, tol=1e-10, max_iter=20, max_polys_per_layer=Inf)
    # nothing to do here
    # no error for linear layer
    return L, 0.
end

function approximate_polynomial(L::OXP.ONNXConv, bounds, degree; cheby=true, verbosity=0, selection=:contiguous, tol=1e-10, max_iter=20, max_polys_per_layer=Inf)
    # nothing to do here
    return L, 0.
end

function approximate_polynomial(L::OXP.ONNXBatchNorm, bounds, degree; cheby=true, verbosity=0, selection=:contiguous, tol=1e-10, max_iter=20, max_polys_per_layer=Inf)
    # batchnorm is already linear
    return L, 0.
end

function VeryDiff.approximate_polynomial(L::OXP.ONNXReshape, bounds, degree; cheby=true, verbosity=0, selection=:contiguous, tol=1e-10, max_iter=20, max_polys_per_layer=Inf)
    return L, 0
end

function approximate_polynomial(L::OXP.ONNXRelu, bounds, degree; cheby=true, verbosity=0, selection=:contiguous, tol=1e-10, max_iter=20, max_polys_per_layer=Inf)
    println("size(bounds) = ", size(bounds))
    in_size = tuple(size(bounds)[1:end-1]..., 1)
    
    lower = vec(selectdim(bounds, ndims(bounds), 1))
    upper = vec(selectdim(bounds, ndims(bounds), 2))

    if max_polys_per_layer == 1
        lower = [minimum(lower)]
        upper = [maximum(upper)]
    end

    res = VeryDiff.approx_relu_poly.(lower, upper, degree, max_iter=max_iter, selection=selection, cheby=cheby, tol=tol)
    ps = hcat(getindex.(res, 1)...)'  # TODO: is there a better way to do vec of vec to matrix?
    ϵs = getindex.(res, 2)  # store them in polynomial layer  # TODO: add error of piecewise polynomial approximation!

    max_idx = argmax(ϵs)
    verbosity > 0 && println("max error = ", ϵs[max_idx], " at idx ", max_idx, " with bounds ", lower[max_idx], " ", upper[max_idx])

    if max_polys_per_layer == 1
        # repeat the single polynomial for all neurons
        # (needed in current implementation of ChebyshevPoly for correct evaluation)
        n_neurons = prod(in_size)
        ϵs = repeat(ϵs[1:1], n_neurons)
        ps = repeat(ps[1:1, :], n_neurons, 1)
        lower = repeat(lower[1:1], n_neurons)
        upper = repeat(upper[1:1], n_neurons)
    end

    # We want the networks to be isomorphic and want to be able to recognize that purely from the node names.
    # Therefore, we reuse the same node names.
    layer = cheby ? ONNXChebyshevPoly(L.inputs, L.outputs, L.name, Matrix(ps), lower, upper, ϵs) : ONNXMonomialPoly(L.inputs, L.outputs, L.name, Matrix(ps))
    return layer, ϵs
end


function approximate_polynomial(L::OXP.ONNXGelu, bounds, degree; cheby=true, verbosity=0, selection=:contiguous, tol=1e-10, max_iter=20, max_polys_per_layer=Inf)
    @assert cheby "Currently, we only allow approximation in Chebyshev coefficients for GeLU layers!"
    
    println("size(bounds) = ", size(bounds))
    # true input size of the layer, without the last dimension containing lower and upper bounds
    # not sure if we need this anywhere ...
    in_size = tuple(size(bounds)[1:end-1]..., 1)

    # lower = @view bounds[:,1]
    # upper = @view bounds[:,2]
    # for convolutional layers, we don't have vector bounds, but tensors of shape (..., 2) where the last dim contains the lower and upper bounds.
    # we need to reshape to vectors for fitting the polynomials.
    lower = vec(selectdim(bounds, ndims(bounds), 1))
    upper = vec(selectdim(bounds, ndims(bounds), 2))

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
        # TODO: better idea than this hack???
        # equal approximation bounds lead to issues with normalization later on, so widen a little bit
        eq_mask = lower .== upper 
        lower[eq_mask] .-= 1e-6
        upper[eq_mask] .+= 1e-6

        if APPROX_POLY_THREADS[] > 1
            res = ThreadsX.map((l, u) -> VeryDiff.approx_gelu_poly(l, u, degree, max_iter=max_iter, selection=selection, cheby=cheby, tol=tol), lower, upper)
        else
            res = approx_gelu_poly.(lower, upper, degree, max_iter=max_iter, selection=selection, cheby=cheby, tol=tol)
        end

        ps = hcat(getindex.(res, 1)...)'  # TODO: is there a better way to do vec of vec to matrix?
        ϵs = getindex.(res, 2)  # TODO: add error of piecewise polynomial approximation!

        # looks cleaner than the above, but consumes WAY to much memory!
        #ps_vecvec, ϵs = zip(res...)
        #ps = stack(ps_vecvec)'

        max_idx = argmax(ϵs)
        verbosity > 0 && println("max error = ", ϵs[max_idx], " at idx ", max_idx, " with bounds ", lower[max_idx], " ", upper[max_idx])
    end

    if max_polys_per_layer == 1
        # repeat the single polynomial for all neurons
        # (needed in current implementation of ChebyshevPoly for correct evaluation)
        n_neurons = prod(in_size)
        ϵs = repeat(ϵs[1:1], n_neurons)
        ps = repeat(ps[1:1, :], n_neurons, 1)
        lower = repeat(lower[1:1], n_neurons)
        upper = repeat(upper[1:1], n_neurons)
    end

    # We want the networks to be isomorphic and want to be able to recognize that purely from the node names.
    # Therefore, we reuse the same node names.
    layer = cheby ? ONNXChebyshevPoly(L.inputs, L.outputs, L.name, Matrix(ps), lower, upper, ϵs) : ONNXMonomialPoly(L.inputs, L.outputs, L.name, Matrix(ps))
    return layer, ϵs
end


function approximate_polynomial(L::ONNXPoly, bounds, degree; cheby=true, verbosity=0, selection=:contiguous, tol=1e-10, max_iter=20, max_polys_per_layer=Inf)
    @warn "skipping already polynomial layer (but degree might differ!/Cheby vs. Monomial Form might not match!)"
    # no error for already polynomial layer
    return L, 0.
end

function approximate_polynomial(L::OXP.ONNXFlatten, bounds, degree; cheby=true, verbosity=0, selection=:contiguous, tol=1e-10, max_iter=20, max_polys_per_layer=Inf)
    # no error for flattening layer
    return L, 0. 
end


# function approximate_polynomial(net::LayeredModel, bounds::AbstractVector, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
#     # attention: bounds are bounds AFTER the layer
#     # for ReLU layers, we need the bound after the linear layer before that, the linear layers don't need any bounds
#     bounds = [[[]]; bounds[1:end-1]]
#     layers = map(x -> approximate_polynomial(x[1], x[2], degree, cheby=cheby, verbosity=verbosity, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer)[1],
#                  zip(net.layers, bounds))
#     return LayeredModel(layers)
# end


"""
Iteratively approximate each layer in the network by a polynomial of a given degree.

The input ranges for the approximation are verified bounds computed by zonotope propagation.

Note: This is not an efficient implementation - it should only be used for testing. 
Can be made more efficient, once there is a convenient method for zonotope propagation through a single NN.

args:
    net - Network to approximate 
    input_lb - concrete lower bounds on the input neurons
    input_ub - concrete upper bounds on the input neurons 
    degree - degree of polynomial approximation for ReLU layers

kwargs:
    verbosity - verbosity level (0: silent, 1: print first 5 lower and upper bounds)
    cheby - whether to use Chebyshev basis (true) or Monomial basis (false) for polynomial approximation
    max_iter - maximum number of iterations for Remez algorithm
    max_polys_per_layer - maximum number of different polynomials to use per layer
"""
function approximate_polynomial_iterative(model::OnnxNet, input_lb::AbstractVector, input_ub::AbstractVector, degree::Integer; verbosity=0, selection=:contiguous, cheby=true, tol=1e-10, max_iter=20, max_polys_per_layer=Inf)
    @assert (max_polys_per_layer == Inf) || (max_polys_per_layer == 1) "only max_polys_per_layer=1 (one polynomial for all neurons) or Inf (one polynomial for each neuron) supported currently"

    # TODO: terrible hack, but without GeminiNetwork we'd have to do all of the initialisation ourselves
    ∂model = GeminiNetwork(model, deepcopy(model));
    input_center = 0.5 .* (input_lb .+ input_ub)
    input_radius = 0.5 .* (input_ub .- input_lb)
    task = VerificationTask(input_center, input_radius, findall(input_radius .!= 0), nothing, nothing, nothing, nothing, nothing, Inf, 1.0)
    P = PropState(true)
    prepare_prop_state!(P, task)

    Zin = P.zono_storage.zonotopes[1].zonotope
    P = propagate!(∂model, P)
    Zout = P.zono_storage.zonotopes[end].zonotope

    bnds = zono_bounds(Zout.∂Z)
    
    layers = Vector{DiffLayer}()
    bounds_layer = [input_lb input_ub]
    for i in 1:length(∂model.diff_layers)
        layer = ∂model.diff_layers[i]
        l1 = get_layer1(layer)

        verbosity > 0 && println("--- layer $i ($(l1.name)) ---")
        verbosity > 0 && println("lower = ", bounds_layer[:,1][1:min(size(bounds_layer, 1), 5)])
        verbosity > 0 && println("upper = ", bounds_layer[:,2][1:min(size(bounds_layer, 1), 5)])
        !all(isfinite.(bounds_layer)) && println("lb non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,1])])
        !all(isfinite.(bounds_layer)) && println("ub non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,2])])

        layer_poly, ϵs = approximate_polynomial(l1, bounds_layer, degree, cheby=cheby, verbosity=verbosity, selection=selection, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer, tol=tol)
        diff_layer_poly = DiffLayer(layer.layer_idx, layer.inputs, layer.outputs, layer_poly, layer_poly, layer.layer2)
        push!(layers, diff_layer_poly)

        ∂model_cur = GeminiNetwork(∂model.inputs, layers)
        P = PropState(true)
        prepare_prop_state!(P, task)
        init_bounds_cache_approximation_domain!(∂model_cur, P)
        P = propagate!(∂model_cur, P)
        
        Zout = P.zono_storage.zonotopes[end].zonotope
        bounds_layer = zono_bounds(Zout.Z₁)
    end

    model_poly = deepcopy(model)
    for l in layers
        l1 = get_layer1(l)
        model_poly.nodes[l1.name] = l1
    end

    return model_poly
end


function approximate_polynomial_iterative_sampling(net::OnnxNet{S}, X_in::AbstractVector, degree; widen_factor=2., selection=:contiguous, verbosity=0, tol=1e-10, cheby=true, max_iter=20, max_polys_per_layer=Inf) where S
    @assert (max_polys_per_layer == Inf) || (max_polys_per_layer == 1) "only max_polys_per_layer=1 (one polynomial for all neurons) or Inf (one polynomial for each neuron) supported currently"
    layers_poly = Vector{OXP.Node{S}}()

    net_layers, io_map = Definitions.sort_network(net)

    output_data = Dict{S, AbstractArray}()
    output_data[OXP.get_input_names(net)[1]] = X_in

    for l in net_layers
        inputs = OXP.collect_inputs(net, l.node.name, output_data)

        @assert length(inputs) == 1 "Currently only single input layers are supported for sampling-based approximation"
        ys_layer = inputs[1]
        # stack along the last dimension --> (size(ys_layer)..., n_samples)
        Y_layer = stack(ys_layer)

        @show size(ys_layer)
        @show size(Y_layer)

        lb_layer = vec(minimum(Y_layer, dims=ndims(Y_layer)))
        ub_layer = vec(maximum(Y_layer, dims=ndims(Y_layer)))

        center = 0.5 .* (lb_layer .+ ub_layer)
        radius = 0.5 .* (ub_layer .- lb_layer)
        widen_radius = widen_factor .* radius
        lb_layer_widened = center .- widen_radius
        ub_layer_widened = center .+ widen_radius

        bounds_layer = hcat(lb_layer_widened, ub_layer_widened)

        verbosity > 0 && println("--- layer ", l.node.name, " ---")
        verbosity > 0 && println("lower = ", bounds_layer[:,1][1:min(size(bounds_layer, 1), 5)])
        verbosity > 0 && println("upper = ", bounds_layer[:,2][1:min(size(bounds_layer, 1), 5)])
        !all(isfinite.(bounds_layer)) && println("lb non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,1])])
        !all(isfinite.(bounds_layer)) && println("ub non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,2])])

        layer_poly, ϵs = approximate_polynomial(l.node, bounds_layer, degree, cheby=cheby, verbosity=verbosity, selection=selection, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer, tol=tol)
        push!(layers_poly, layer_poly)

        outputs = [OXP.onnx_node_to_flux_layer(layer_poly)(y) for y in ys_layer]
        out_names = net.nodes[layer_poly.name].outputs
        @assert length(out_names) == 1 "Currently only single output layers are supported for sampling-based approximation"
        output_data[out_names[1]] = outputs
    end

    model = deepcopy(net)
    # need to return a full OnnxNet here for later steps, but replace the original nodes with the polynomial approximations.
    # structure of the model did not change, so we can just update model.nodes
    for layer_poly in layers_poly
        model.nodes[layer_poly.name] = layer_poly
    end

    return model
end


"""
Iteratively approximate each layer in the network by a polynomial of a given degree with pre-activation ranges computed via alpha-beta-CROWN.

The pre-activation ranges are computed using additional error inputs added to the network after every activation layer.

args:
    onnx_path - Path to ONNX network to approximate
    degree - degree of polynomial approximation for ReLU layers

kwargs:
    verbosity - verbosity level (0: silent, 1: print first 5 lower and upper bounds)
    input_bounds - dictionary input_name => (lower_bound, upper_bound) to use for pre-activation bound computation 
                   (if not provided, defaults to [0,1] bounds for all inputs). Bounds need to be in correct shape!
    cheby - whether to use Chebyshev basis (true) or Monomial basis (false) for polynomial approximation
    max_iter - maximum number of iterations for Remez algorithm
    max_polys_per_layer - maximum number of different polynomials to use per layer
    tight_gelu - use tight initialization of gelu relaxation
"""
function approximate_polynomial_abcrown(onnx_path, degree; input_bounds=nothing, cheby=true, verbosity=0, tol=1e-10, selection=:contiguous, max_iter=20, max_polys_per_layer=Inf, tight_gelu=true)
    ERROR_NODES_SCRIPT = pyimport("insert_error_nodes")
    BOUNDS_SCRIPT      = pyimport("get_bounds")

    if isnothing(input_bounds)
        input_bounds = Dict()
    end

    # load model in julia
    model = load_onnx_model(onnx_path)
    net_layers, io_map = Definitions.sort_network(model)

    # extend network by error inputs
    error_net_path = "error_net.onnx"
    ERROR_NODES_SCRIPT.insert_error_nodes(onnx_path, error_net_path)
    # run(`$(ABCROWN_PYTHONPATH) $(ERROR_NODES_SCRIPT) $(onnx_path) $(error_net_path)`)

    input_bounds_file = "input_bounds.h5"
    activations_considered = []
    error_magnitudes = []
    layer_bounds = []
    layers_poly = Vector{OXP.Node}()

    for (i, l) in enumerate(net_layers)
        if (l.node isa OXP.ONNXRelu) || (l.node isa OXP.ONNXGelu)
            # prepare input bounds
            h5open(input_bounds_file, "w") do file 
                for (k, v) in model.input_shapes
                    if k in keys(input_bounds)
                        lb, ub = input_bounds[k]
                        @assert size(lb) == size(ub) == v "input bounds for $k have incorrect shape, expected $v but got $(size(lb)) and $(size(ub))"
                    else
                        lb = zeros(v)
                        ub = ones(v)
                    end
                    # hdf5 already converts from WHCN to NCHW
                    file[k] = cat(lb, ub, dims=ndims(lb))
                end

                for (l, ϵs) in zip(activations_considered, error_magnitudes)
                    error_name = "eps_" * l.node.name
                    file[error_name] = reshape(ϵs, :, 1)  # append a batch dimension
                end
            end

            output_names = l.node.inputs
            @assert length(output_names) == 1 "Currently only single output activation layers are supported for alpha-beta-CROWN-based approximation"
            output_name = output_names[1]
            output_bounds = "out_bounds.h5"
            BOUNDS_SCRIPT.compute_pre_activation_bounds(error_net_path, input_bounds_file, output_name; outfile=output_bounds, method="alpha-crown", tight_gelu=tight_gelu)

            bounds = h5open(output_bounds, "r") do file 
                read(file[output_name])
            end

            layer_poly, ϵs = VeryDiff.approximate_polynomial(l.node, bounds, degree, cheby=cheby, verbosity=verbosity, selection=selection, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer, tol=tol)

            push!(activations_considered, l)
            push!(error_magnitudes, ϵs)
            push!(layer_bounds, bounds)

            if (verbosity > 0) || !all(isfinite.(bounds))
                lower_print = vec(selectdim(bounds, ndims(bounds), 1))
                upper_print = vec(selectdim(bounds, ndims(bounds), 2))

                if verbosity > 0
                    println("--- layer $i ---")
                    println("lower = ", lower_print[1:min(length(lower_print), 5)])
                    println("upper = ", upper_print[1:min(length(upper_print), 5)])
                end 

                if !all(isfinite.(bounds))
                    println("lb non-finite: ", (1:length(lower_print))[.~isfinite.(lower_print)])
                    println("ub non-finite: ", (1:length(upper_print))[.~isfinite.(upper_print)])
                end
            end
        else
            layer_poly = l.node
        end
        push!(layers_poly, layer_poly)
    end

    # need to return a full OnnxNet here for later steps, but replace the original nodes with the polynomial approximations.
    # structure of the model did not change, so we can just update model.nodes
    for layer_poly in layers_poly
        model.nodes[layer_poly.name] = layer_poly
    end

    return model
end
