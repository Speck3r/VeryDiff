using VeryDiff, LinearAlgebra, VNNLib
import VeryDiff: approximate_polynomial_iterative, approximate_polynomial


function get_empirical_bounds(net::Network, z::Zonotope; n_inputs=1000)
    bounds = []

    for i in 1:n_inputs
        x = VeryDiff.random_point(z)
        ys = VeryDiff.intermediate_activations(net, x)

        if length(bounds) == 0
            bounds = [[y y] for y in ys]
        else
            for j in 1:length(bounds)
                bounds[j][:,1] .= min.(bounds[j][:,1], ys[j])
                bounds[j][:,2] .= max.(bounds[j][:,2], ys[j])
            end
        end
    end

    return bounds 
end


function widen_bounds(bnds::AbstractVector, factor::N) where N<:Number
    bnd_new = deepcopy(bnds)
    for i in 1:length(bnds) 
        bnd = bnds[i]
        radius = 0.5 .* (bnd[:,2] .- bnd[:,1])
        center = 0.5 .* (bnd[:,1]  .+ bnd[:,2])

        bnd_new[i][:,1] .= center .- factor .* radius
        bnd_new[i][:,2] .= center .+ factor .* radius
    end 

    return bnd_new
end



## Example 

model_file = "./test/examples/networks/mnist-net_256x4.onnx"
poly_dir = "./test/examples/poly_coeffs/models/mnist_fc/poly_2/"

z = Zonotope(I(784) .* 0.05, zeros(784), I(784));

nn = VNNLib.load_network(model_file)

test_set = [VeryDiff.random_point(z) for _ in 1:10000]
ys = [nn(x) for x in test_set]
Y  = hcat(ys...)';


println("## Zono bounds")
maes_zono = []
for degree in 1:30
    nn_poly = approximate_polynomial_iterative(nn, z, degree, verbosity=0, cheby=true)
    ys_poly = [nn_poly(x) for x in test_set]
    Y_poly = hcat(ys_poly...)'

    mse = sum(sum((Y .- Y_poly).^2, dims=2)) / size(Y, 1)
    mae = maximum(abs.(Y .- Y_poly))
    println("degree = ", degree, " - mse = ", mse, " - mae = ", mae)
    push!(maes_zono, mae)
end

bnds10k = get_empirical_bounds(nn, z, n_inputs=10000)
println("\n## Sampled bounds")
maes_sampled = []
for degree in 1:30
    nn_poly = approximate_polynomial(nn, bnds10k, degree)
    ys_poly = [nn_poly(x) for x in test_set]
    Y_poly = hcat(ys_poly...)'

    mse = sum(sum((Y .- Y_poly).^2, dims=2)) / size(Y, 1)
    mae = maximum(abs.(Y .- Y_poly))
    println("degree = ", degree, " - mse = ", mse, " - mae = ", mae)
    push!(maes_sampled, mae)
end

println("\n## Widened bounds 2")
wbnds = widen_bounds(bnds10k, 2)
maes_widen2 = []
for degree in 1:30
    nn_poly = approximate_polynomial(nn, wbnds, degree)
    ys_poly = [nn_poly(x) for x in test_set]
    Y_poly = hcat(ys_poly...)'

    mse = sum(sum((Y .- Y_poly).^2, dims=2)) / size(Y, 1)
    mae = maximum(abs.(Y .- Y_poly))
    println("degree = ", degree, " - mse = ", mse, " - mae = ", mae)
    push!(maes_widen2, mae)
end

println("\n## Widened bounds 4")
wbnds = widen_bounds(bnds10k, 4)
maes_widen4 = []
for degree in 1:30
    nn_poly = approximate_polynomial(nn, wbnds, degree)
    ys_poly = [nn_poly(x) for x in test_set]
    Y_poly = hcat(ys_poly...)'

    mse = sum(sum((Y .- Y_poly).^2, dims=2)) / size(Y, 1)
    mae = maximum(abs.(Y .- Y_poly))
    println("degree = ", degree, " - mse = ", mse, " - mae = ", mae)
    push!(maes_widen4, mae)
end


plot(maes_zono, yaxis=:log, label="zono", xlabel="approximation degree", ylabel="MAE")
plot!(maes_sampled[1:7], label="sampled", linestyle=:dash)
plot!(maes_widen2, label="widen2")
plot!(maes_widen4, label="widen4")




prop_state = PropState(true)

ẑ = nn(z, prop_state)
bounds_relu = zono_bounds(ẑ)

for degree in 1:10
    nn_poly = approximate_polynomial_iterative(nn, z, degree)

    ẑ_poly = nn_poly(z, prop_state)
    bounds_poly = zono_bounds(ẑ_poly)

    # propagate differential zonotope through the difference network
    nn_diff = GeminiNetwork(nn_poly, nn);
    ∂z = Zonotope(zeros(784, 784), zeros(784), nothing)
    zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)
    ẑΔ = nn_diff(zΔ, PropState(true))

    bounds_diff = zono_bounds(ẑΔ.∂Z)

    println("ReLU:\tupper = ", bounds_relu[:,2])
    println("Poly ($degree):\tupper = ", bounds_poly[:,2])
    println("Diff ($degree):\tupper = ", bounds_diff[:,2])
end





# debugging
prop_state = VeryDiff.PropState(true)
layers_poly = []
push!(layers_poly, nn.layers[1])  # start with linear layer already in the list
bound = nothing 

layer2 = nn.layers[2]
net_partial = Network(layers_poly)

ẑ = net_partial(z, prop_state)
bnds = zono_bounds(ẑ)

layer2_poly = approximate_polynomial(layer2, bnds, 20, cheby=true)
push!(layers_poly, layer2_poly);

push!(layers_poly, nn.layers[3])  # another linear layer
layer4 = nn.layers[4]
net_partial = Network(layers_poly)
ẑ = net_partial(z, prop_state)
bnds = zono_bounds(ẑ)

layer4_poly = approximate_polynomial(layer2, bnds, 20, cheby=true)


idx = 1
fp = x -> VeryDiff.clenshaw_chebyshev(layer4_poly.coeffs[idx,:], x, layer4_poly.l[idx], layer4_poly.u[idx])
xs = range(bnds[idx,1], bnds[idx,2], 300)
plot(xs, max.(0, xs), label="relu", framestyle=:origin)
plot!(xs, fp.(xs), label="p")


α, β, ϵ = VeryDiff.approx_polynomial_lin(layer2_poly.coeffs[idx,:], bnds[idx,1], bnds[idx,2], bnds[idx,1], bnds[idx,2]);
fp_lin = x -> VeryDiff.clenshaw_chebyshev([β, α], x, bnds[idx,1], bnds[idx,2])
plot!(xs, fp_lin.(xs), label="lin")

α, β, γ = VeryDiff.get_linear_relaxation(layer2_poly, bnds[:,1], bnds[:,2]);

plot!(xs, α[idx] .* xs .+ β[idx], label="lin_mono")
plot!(xs, α[idx] .* xs .+ β[idx] .+ γ[idx], label="up_mono", linestyle=:dash)
plot!(xs, α[idx] .* xs .+ β[idx] .- γ[idx], label="lo_mono", linestyle=:dash)







cs, err = VeryDiff.approx_relu_poly(bnds[idx,1], bnds[idx,2], 15, max_iter=5, cheby=true)
fp = x -> VeryDiff.clenshaw_chebyshev(cs, x, bnds[idx,1], bnds[idx,2])
plot(xs, max.(0, xs), label="relu", framestyle=:origin)
plot!(xs, fp.(xs), label="p")

cs_norm = VeryDiff.normalize_chebyshev(cs, bnds[idx,1], bnds[idx,2])
fp_norm = x -> VeryDiff.clenshaw_chebyshev(cs_norm, x);

plot!(xs, fp_norm.(xs), label="p_norm")


xs = range(-6, 6, 300)
#plot(xs, fp.(xs), label="p")
#plot!(xs, fp_norm.(xs), label="p_norm")
plot(xs, fp.(xs) .- fp_norm.(xs), label="p - p_norm")