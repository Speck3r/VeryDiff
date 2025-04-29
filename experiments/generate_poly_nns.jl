
using VeryDiff, VNNLib, CSV, Plots, LinearAlgebra, JLD2, Dates
import VeryDiff: approximate_polynomial_iterative, approximate_polynomial


"""
Given a network and a zonotopic input set, sample random inputs and return the minimal and maximal activation and preactivation values.

args:
    net - network to get bounds from
    z - zonotopic input set

kwargs:
    n_inputs - number of random inputs to sample

returns:
    bounds - list of (n_neurons x 2)-array for each layer holding lower and upper bounds for each neuron 
             after that layer was applied
"""
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


"""
Given a network and a set of inputs, return the minimal and maximal activation and preactivation values.

args:
    net - network to get bounds from
    X_in - set of inputs (vector of vectors)

returns:
    bounds - list of (n_neurons x 2)-array for each layer holding lower and upper bounds for each neuron 
             after that layer was applied
"""
function get_empirical_bounds(net::Network, X_in::AbstractVector)
    bounds = []

    for x in X_in
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


"""
Given an array of lower and upper bounds for each neuron, widen the bounds by a given factor.

If you have lower and upper bounds l, u for a neuron, the new bounds are given by

    l_new = center - factor * radius
    u_new = center + factor * radius

where center = 0.5 * (l + u) and radius = 0.5 * (u - l).

args:
    bnds - list of (n_neurons x 2)-array for each layer holding lower and upper bounds for each neuron 
           after that layer was applied
    factor - factor to widen the bounds by

returns:
    bnd_new - list of (n_neurons x 2)-array for each layer holding widened lower and upper bounds for each neuron 
              after that layer was applied
"""
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


mae_fun(y, ŷ) = sum(maximum(abs.(y .- ŷ), dims=2)) / size(y, 1)
mse_fun(y, ŷ) = sum(sum((y .- ŷ).^2, dims=2)) / size(y, 1)

"""
Accuracy function for classification tasks.

args:
    y - vector of true labels (make sure they are 1 indexed)
    ŷ - matrix (n_inputs × outputs) of predicted logits
"""
function acc_fun(y::AbstractArray{<:Number}, ŷ::AbstractArray)
    # argmax(ŷ, dims=2) returns a CartesianIndex object, so we need to extract the index of the row
    sum(getindex.(argmax(ŷ, dims=2), 2) .== y) / size(y, 1)
end



function generate_poly_network(net::Network, z::Zonotope, degree; bounds=nothing, X_test=nothing, y_test=nothing, y_labels=nothing, empirical=false, cheby=true, verbosity=0, max_iter=20, widen_factor=1.0)
    if empirical
        bounds = isnothing(bounds) ? get_empirical_bounds(net, z) : bounds
        bounds = widen_bounds(bounds, widen_factor)
        nn_poly = approximate_polynomial(net, bounds,degree)
    else
        nn_poly = approximate_polynomial_iterative(net, z, degree, cheby=cheby, verbosity=verbosity, max_iter=max_iter)
    end

    if !isnothing(X_test) && !isnothing(y_test)
        ŷ = [nn_poly(x) for x in X_test]
        ŷ = hcat(ŷ...)'
        mae = mae_fun(y_test, ŷ)
        mse = mse_fun(y_test, ŷ)
        # need to add one as labels are 0 indexed
        acc = acc_fun(y_labels .+ 1, ŷ)
        return nn_poly, mae, mse, acc
    else
        return nn_poly
    end   
end


f_mnist = CSV.File(string(@__DIR__, "/data/mnist_train.csv"), header=false)
X_test = [Float64.([x for x in f_mnist[i]][2:end]) ./ 255 for i in 1:size(f_mnist, 1)]
y_test = [[x for x in f_mnist[i]][1] for i in 1:size(f_mnist, 1)]

model_file = string(@__DIR__, "/../test/examples/networks/mnist-net_256x4.onnx")
net = VNNLib.load_network(model_file)

bnds_mnist = get_empirical_bounds(net, X_test);

ŷ = [net(x) for x in X_test]
ŷ = hcat(ŷ...)'
acc_fun(y_test .+ 1, ŷ)

z = Zonotope(I(784) .* 0.5, zeros(784) .+ 0.5, I(784))

z_out = net(z, PropState(true));

println("===== MNIST 4x256 Iterative =====")
maes = []
mses = []
accs = []
times = []
nets = []
for degree in (1:9) ∪ (10:10:200)
    t = @elapsed nn_poly, mae, mse, acc = generate_poly_network(net, z, degree, X_test=X_test, y_test=ŷ, y_labels=y_test, empirical=false, verbosity=1, max_iter=20)
    println("degree = ", degree, " mae = ", mae, " mse = ", mse, " acc = ", acc, " (", t, "s)")
    push!(maes, mae)
    push!(mses, mse) 
    push!(accs, acc)
    push!(times, t)
    push!(nets, nn_poly)
end

jldsave(string("mnist_4x256_zono_polys_data_", now(), ".jld2"); nets, maes, mses, accs, times)

model_file = string(@__DIR__, "/networks/mnist_256x4_2e5.onnx")
net = VNNLib.load_network(model_file)

z = Zonotope(I(784) .* 0.5, zeros(784) .+ 0.5, I(784))

println("===== MNIST L1 4x256 Iterative =====")
maes_l1 = []
mses_l1 = []
accs_l1 = []
times_l1 = []
nets_l1 = []
for degree in (1:9) ∪ (10:10:200)
    t = @elapsed nn_poly, mae, mse, acc = generate_poly_network(net, z, degree, X_test=X_test, y_test=ŷ, y_labels=y_test, empirical=false, verbosity=1, max_iter=20)
    println("degree = ", degree, " mae = ", mae, " mse = ", mse, " acc = ", acc, " (", t, "s)")
    push!(maes_l1, mae)
    push!(mses_l1, mse) 
    push!(accs_l1, acc)
    push!(times_l1, t)
    push!(nets_l1, nn_poly)
end

jldsave(string("mnist_4x256_2e5_zono_polys_data_", now(), ".jld2"); nets_l1, maes_l1, mses_l1, accs_l1, times_l1)


model_file = string(@__DIR__, "/networks/mnist_256x4_1e4.onnx")
net = VNNLib.load_network(model_file)

ŷ = [net(x) for x in X_test]
ŷ = hcat(ŷ...)'
acc_fun(y_test .+ 1, ŷ)

z = Zonotope(I(784) .* 0.5, zeros(784) .+ 0.5, I(784))

println("===== MNIST L1 1e4 4x256 Iterative =====")
maes_l1_1e4 = []
mses_l1_1e4 = []
accs_l1_1e4 = []
times_l1_1e4 = []
nets_l1_1e4 = []
for degree in (1:9) ∪ (10:10:200)
    t = @elapsed nn_poly, mae, mse, acc = generate_poly_network(net, z, degree, X_test=X_test, y_test=ŷ, y_labels=y_test, empirical=false, verbosity=1, max_iter=20)
    println("degree = ", degree, " mae = ", mae, " mse = ", mse, " acc = ", acc, " (", t, "s)")
    push!(maes_l1_1e4, mae)
    push!(mses_l1_1e4, mse) 
    push!(accs_l1_1e4, acc)
    push!(times_l1_1e4, t)
    push!(nets_l1_1e4, nn_poly)
end

jldsave(string("mnist_4x256_1e4_zono_polys_data_", now(), ".jld2"); nets_l1_1e4, maes_l1_1e4, mses_l1_1e4, accs_l1_1e4, times_l1_1e4)



plot((1:9) ∪ (10:10:200),  maes, label="maes", marker=:diamond, xlabel="degree", ylabel="MAE", yaxis=:log, title="MNIST 4x256")
plot!((1:9) ∪ (10:10:200),  maes_l1, label="maes L1", marker=:diamond)

plot((1:9) ∪ (10:10:200),  mses, label="mses", marker=:diamond, xlabel="degree", ylabel="MSE", yaxis=:log, title="MNIST 4x256")
plot!((1:9) ∪ (10:10:200),  mses_l1, label="mses L1", marker=:diamond)

plot((1:9) ∪ (10:10:200),  accs, label="acc", marker=:diamond, xlabel="degree", ylabel="accuracy", title="MNIST 4x256", legend=:inside)
plot!((1:9) ∪ (10:10:200),  accs_l1, label="acc L1", marker=:diamond)

plot((1:9) ∪ (10:10:200),  times, label="t", marker=:diamond, xlabel="degree", ylabel="time (sec)", title="MNIST 4x256")
plot!((1:9) ∪ (10:10:200),  times_l1, label="t L1", marker=:diamond)


##############################################
## MNIST 256x6 
##############################################


model_file = string(@__DIR__, "/../test/examples/networks/mnist-net_256x6.onnx")
net = VNNLib.load_network(model_file)


println("===== MNIST 6x256 Iterative =====")
maes6 = []
mses6 = []
accs6 = []
times6 = []
nets6 = []
for degree in (1:9) ∪ (10:10:200)
    t = @elapsed nn_poly, mae, mse, acc = generate_poly_network(net, z, degree, X_test=X_test, y_test=ŷ, y_labels=y_test, empirical=false, verbosity=1, max_iter=20)
    println("degree = ", degree, " mae = ", mae, " mse = ", mse, " acc = ", acc, " (", t, "s)")
    push!(maes6, mae)
    push!(mses6, mse) 
    push!(accs6, acc)
    push!(times6, t)
    push!(nets6, nn_poly)
end

jldsave(string("mnist_6x256_zono_polys_data_", now(), ".jld2"); nets_l1, maes_l1, mses_l1, accs_l1, times_l1)

model_file = string(@__DIR__, "/networks/mnist_256x6_2e5.onnx")
net = VNNLib.load_network(model_file)

println("===== MNIST L1 6x256 Iterative =====")
maes6_l1 = []
mses6_l1 = []
accs6_l1 = []
times6_l1 = []
nets6_l1 = []
for degree in (1:9) ∪ (10:10:200)
    t = @elapsed nn_poly, mae, mse, acc = generate_poly_network(net, z, degree, X_test=X_test, y_test=ŷ, y_labels=y_test, empirical=false, verbosity=1, max_iter=20)
    println("degree = ", degree, " mae = ", mae, " mse = ", mse, " acc = ", acc, " (", t, "s)")
    push!(maes6_l1, mae)
    push!(mses6_l1, mse) 
    push!(accs6_l1, acc)
    push!(times6_l1, t)
    push!(nets6_l1, nn_poly)
end

jldsave(string("mnist_6x256_2e5_zono_polys_data_", now(), ".jld2"); nets_l1, maes_l1, mses_l1, accs_l1, times_l1)


plot((1:9) ∪ (10:10:200),  maes6, label="maes", marker=:diamond, xlabel="degree", ylabel="MAE", yaxis=:log, title="MNIST 6x256")
plot!((1:9) ∪ (10:10:200),  maes6_l1, label="maes L1", marker=:diamond)

plot((1:9) ∪ (10:10:200),  mses6, label="mses", marker=:diamond, xlabel="degree", ylabel="MSE", yaxis=:log, title="MNIST 6x256")
plot!((1:9) ∪ (10:10:200),  mses6_l1, label="mses L1", marker=:diamond)

plot((1:9) ∪ (10:10:200),  accs6, label="acc", marker=:diamond, xlabel="degree", ylabel="accuracy", title="MNIST 6x256")
plot!((1:9) ∪ (10:10:200),  accs6_l1, label="acc L1", marker=:diamond)

plot((1:9) ∪ (10:10:200),  times6, label="t", marker=:diamond, xlabel="degree", ylabel="time (sec)", title="MNIST 6x256")
plot!((1:9) ∪ (10:10:200),  times6_l1, label="t L1", marker=:diamond)



# overall plots for all networks
plot((1:9) ∪ (10:10:200),  maes, label="maes 4x256", marker=:diamond, xlabel="degree", ylabel="MAE", yaxis=:log, title="MNIST", color=1)
plot!((1:9) ∪ (10:10:200),  maes_l1, label="maes L1 4x256", marker=:utriangle, color=1, linestyle=:dash)
plot!((1:9) ∪ (10:10:200),  maes6, label="maes 6x256", marker=:diamond, color=2)
plot!((1:9) ∪ (10:10:200),  maes6_l1, label="maes L1 6x256", marker=:utriangle, color=2, linestyle=:dash)

plot((1:9) ∪ (10:10:200),  mses, label="mses 4x256", marker=:diamond, xlabel="degree", ylabel="MSE", yaxis=:log, title="MNIST", color=1)
plot!((1:9) ∪ (10:10:200),  mses_l1, label="mses L1 4x256", marker=:utriangle, color=1, linestyle=:dash)
plot!((1:9) ∪ (10:10:200),  mses6, label="mses 6x256", marker=:diamond, color=2)
plot!((1:9) ∪ (10:10:200),  mses6_l1, label="mses L1 6x256", marker=:utriangle, color=2, linestyle=:dash)

plot((1:9) ∪ (10:10:200),  accs, label="accs 4x256", marker=:diamond, xlabel="degree", ylabel="accuracy", title="MNIST", color=1, legend=:inside)
plot!((1:9) ∪ (10:10:200),  accs_l1, label="accs L1 4x256", marker=:utriangle, color=1, linestyle=:dash)
plot!((1:9) ∪ (10:10:200),  accs6, label="accs 6x256", marker=:diamond, color=2)
plot!((1:9) ∪ (10:10:200),  accs6_l1, label="accs L1 6x256", marker=:utriangle, color=2, linestyle=:dash)

plot((1:9) ∪ (10:10:200),  times, label="times 4x256", marker=:diamond, xlabel="degree", ylabel="time (sec)", title="MNIST", color=1, legend=:inside)
plot!((1:9) ∪ (10:10:200),  times_l1, label="timess L1 4x256", marker=:utriangle, color=1, linestyle=:dash)
plot!((1:9) ∪ (10:10:200),  times6, label="times 6x256", marker=:diamond, color=2)
plot!((1:9) ∪ (10:10:200),  times6_l1, label="times L1 6x256", marker=:utriangle, color=2, linestyle=:dash)


bnds = get_empirical_bounds(net, z, n_inputs=10000)

println("===== MNIST 4x256 Empirical Widen 4x =====")
maes = []
mses = []
accs = []
times = []
nets = []
for degree in 1:100
    t = @elapsed nn_poly, mae, mse, acc = generate_poly_network(net, z, degree, bounds=bnds, X_test=X_test, y_test=ŷ, y_labels=y_test, empirical=true, verbosity=1, widen_factor=4.0)
    println("degree = ", degree, " mae = ", mae, " mse = ", mse, " acc = ", acc, " (", t, "s)")
    push!(maes, mae)
    push!(mses, mse) 
    push!(accs, acc)
    push!(times, t)
    push!(nets, nn_poly)
end



#################################
### Verified Equivalence      ###
#################################

degrees = (1:9) ∪ (10:10:200)

res = load("./mnist_4x256_1e4_zono_polys_data_2025-03-27T17:39:57.751.jld2")
nets_l1_1e4 = res["nets_l1_1e4"];
maes_l1_1e4 = res["maes_l1_1e4"];

#model_file = string(@__DIR__, "/networks/mnist_256x4_2e5.onnx")
model_file = string(@__DIR__, "/networks/mnist_256x4_1e4.onnx")
nn = VNNLib.load_network(model_file)

VeryDiff.OPTIM_ITERS[] = 0
prop_state = PropState(true)
∂bounds = []
for (i, nn_poly) in enumerate(nets_l1_1e4)
    #ẑ_poly = nn_poly(z, prop_state)
    #bounds_poly = zono_bounds(ẑ_poly)
    #println("\nlbs = ", bounds_poly[:,1])
    #println("ubs = ", bounds_poly[:,2])

    # propagate differential zonotope through the difference network
    nn_diff = GeminiNetwork(nn_poly, nn);

    # need to convert to matrix, s.t. z and ∂z have the same type
    z = Zonotope(Matrix(I(784)) .* 0.5, zeros(784) .+ 0.5, I(784))
    ∂z = Zonotope(zeros(784, 784), zeros(784), nothing)
    zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)
    #@profile ẑΔ = nn_diff(zΔ, PropState(true))
    ẑΔ = nn_diff(zΔ, PropState(true))

    bounds_diff = zono_bounds(ẑΔ.∂Z)
    ∂bound = maximum(abs.(bounds_diff))
    println("\ndegree = ", degrees[i], " ∂bound = ", ∂bound)
    println("\tlbs = ", bounds_diff[:,1])
    println("\tubs = ", bounds_diff[:,2])
    println("")
    push!(∂bounds, ∂bound)
end


#plot(degrees, ∂bounds, label="∂bounds L1", marker=:diamond, xlabel="degree", ylabel="difference", yaxis=:log, title="MNIST 4x256 L1 - Verified Difference")
#plot!(degrees, maes_l1, label="maes L1", marker=:utriangle, linestyle=:dash)

plot(degrees, ∂bounds, label="∂bounds L1", marker=:diamond, xlabel="degree", ylabel="difference", yaxis=:log, title="MNIST 4x256 L1 1e-4 - Verified Difference")
plot!(degrees, maes_l1_1e4, label="maes L1", marker=:utriangle, linestyle=:dash)


bnds_1e4 = get_empirical_bounds(nn, X_test);



y = [nn(x) for x in X_test]
y = hcat(y...)'
ŷ = [nn_poly(x) for x in X_test]
ŷ = hcat(ŷ...)';

d = ∂bounds[end] .* ones(length(y[1,:]))
max_idx = argmax(y[1,:])
d = ifelse.(1:length(y[1,:]) .== max_idx, .-d, d)








# This doesn't work right now because of some NaNs :-(

# Why do I have to write this??? I thought it turns off automatically 
# at Initialization if Gurobi is not available
VeryDiff.USE_GUROBI = false

#nn_poly = nets_l1[end]
nn_poly = nets_l1_1e4[end]
property_check = get_top1_property(naive=false, delta=0.9)
split_heuristic = top1_configure_split_heuristic(1)
verify_network(nn_poly, nn, [zeros(784) ones(784)], property_check, split_heuristic; timeout=600)


## debugging
nn_poly = nets_l1[end]
nn_diff = GeminiNetwork(nn_poly, nn);

# need to convert to matrix, s.t. z and ∂z have the same type
z = Zonotope(Matrix(I(784)) .* 0.5, zeros(784) .+ 0.5, I(784))
∂z = Zonotope(zeros(784, 784), zeros(784), nothing)
zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)
#@profile ẑΔ = nn_diff(zΔ, PropState(true))
ẑΔ = nn_diff(zΔ, PropState(true))

# during porpagation of the differential zonotope, we get max_errors of 
# 0.011874595973571545
# 0.08377719067152946
# 0.48085414516227853
# 0.8577763650704332
#
# but during construction, we get errors of 
# 0.011874595973621282
# 0.03614989671659785
# 0.11536441284661691
# 
# 
# which only match for the initial layer and are much smaller for the later layers!


bounds_diff = zono_bounds(ẑΔ.∂Z)

t = @elapsed nn_poly, mae, mse, acc = generate_poly_network(nn, z, 200, X_test=X_test, y_test=ŷ, y_labels=y_test, empirical=false, verbosity=1, max_iter=20)




res_2e5 = load("./mnist_4x256_2e5_zono_polys_data_2025-03-27T17:39:57.751.jld2");
