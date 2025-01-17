
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


function approximate_polynomial(L::Dense, bounds, degree)
    # nothing to do here
    return L    
end

function approximate_polynomial(L::VNNLib.ReLU, bounds, degree)
    lower = @view bounds[:,1]
    upper = @view bounds[:,2]

    res = VeryDiff.approx_relu_poly.(lower, upper, degree, max_iter=5)
    ps = hcat(getindex.(res, 1)...)'  # TODO: is there a better way to do vec of vec to matrix?
    ϵs = getindex.(res, 2)  # don't really need them, just for debugging 

    #@show maximum(ϵs)

    layer = Poly(Matrix(ps))
    return layer
end


function approximate_polynomial(L::Poly, bounds, degree)
    @warn "skipping already polynomial layer (but degree might differ!)"
    return L    
end


function approximate_polynomial(net::Network, bounds::AbstractVector, degree)
    # attention: bounds are bounds AFTER the layer
    # for ReLU layers, we need the bound after the linear layer before that, the linear layers don't need any bounds
    bounds = [[[]]; bounds[1:end-1]]
    layers = map(x -> approximate_polynomial(x[1], x[2], degree), zip(net.layers, bounds))
    return Network(layers)
end


function approximate_polynomial_iterative(net::Network, input_set, degree; verbosity=0)
    prop_state = PropState(true)
    layers_poly = []
    bound = nothing
    for i in 1:length(net.layers)
        layer = net.layers[i]
        net_partial = Network([layers_poly; layer])

        ẑ = net_partial(input_set, prop_state)
        bound = zono_bounds(ẑ)

        verbosity > 0 && println("lower = ", bound[:,1][1:min(size(bound, 1), 5)])
        verbosity > 0 && println("upper = ", bound[:,2][1:min(size(bound, 1), 5)])

        layer_poly = approximate_polynomial(layer, bound, degree)
        push!(layers_poly, layer_poly)
    end

    return Network(layers_poly)
end
