
"""
For each layer in the network, compute the interval bounds after that layer was applied.

args:
    net - Network to get bounds from 
    input_set - input set for which to get bounds 

returns:
    bounds - list of (n_neurons x 2)-array for each layer holding lower and upper bounds for each neuron 
             after that layer was applied
"""
function get_zono_bounds(net::Network, input_set::Zonotope)
    bounds = []
    
    prop_state = PropState(true)
    for i in 1:length(net.layers)
        layers = net.layers[1:i]
        net_partial = Network(layers)

        ẑ = net_partial(input_set, prop_state)
        bounds_layer = zono_bounds(ẑ)
        push!(bounds, bounds_layer)
    end

    return bounds
end


function approximate_polynomial(L::Dense, bounds, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
    # nothing to do here
    return L    
end

function approximate_polynomial(L::VNNLib.ReLU, bounds, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
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

    layer = cheby ? ChebyshevPoly(Matrix(ps), lower, upper) : MonomialPoly(Matrix(ps))
    return layer
end


function approximate_polynomial(L::Poly, bounds, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
    @warn "skipping already polynomial layer (but degree might differ!/Cheby vs. Monomial Form might not match!)"
    return L    
end


function approximate_polynomial(net::Network, bounds::AbstractVector, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
    # attention: bounds are bounds AFTER the layer
    # for ReLU layers, we need the bound after the linear layer before that, the linear layers don't need any bounds
    bounds = [[[]]; bounds[1:end-1]]
    layers = map(x -> approximate_polynomial(x[1], x[2], degree, cheby=cheby, verbosity=verbosity, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer),
                 zip(net.layers, bounds))
    return Network(layers)
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
function approximate_polynomial_iterative(net, input_set, degree; verbosity=0, cheby=true, max_iter=20, max_polys_per_layer=Inf)
    @assert (max_polys_per_layer == Inf) || (max_polys_per_layer == 1) "only max_polys_per_layer=1 (one polynomial for all neurons) or Inf (one polynomial for each neuron) supported currently"
    prop_state = PropState(true)
    layers_poly = []
    ẑ = input_set
    for i in 1:length(net.layers)
        bounds_layer = zono_bounds(ẑ)

        verbosity > 0 && println("--- layer $i ---")
        verbosity > 0 && println("lower = ", bounds_layer[:,1][1:min(size(bounds_layer, 1), 5)])
        verbosity > 0 && println("upper = ", bounds_layer[:,2][1:min(size(bounds_layer, 1), 5)])
        !all(isfinite.(bounds_layer)) && println("lb non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,1])])
        !all(isfinite.(bounds_layer)) && println("ub non-finite: ", (1:size(bounds_layer,1))[.~isfinite.(bounds_layer[:,2])])

        layer = net.layers[i]
        layer_poly = approximate_polynomial(layer, bounds_layer, degree, cheby=cheby, verbosity=verbosity, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer)
        push!(layers_poly, layer_poly)

        ẑ = layer_poly(ẑ, prop_state)
    end

    return Network(layers_poly)   
end
