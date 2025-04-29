
using VeryDiff, VNNLib, CSV, Plots, LinearAlgebra, JLD2, Dates
import VeryDiff: approximate_polynomial_iterative, approximate_polynomial


σ(x) = 1 / (1 + exp(-x))


mae_fun(y, ŷ) = sum(maximum(abs.(y .- ŷ), dims=2)) / size(y, 1)
mse_fun(y, ŷ) = sum(sum((y .- ŷ).^2, dims=2)) / size(y, 1)

"""
Accuracy function for classification tasks.

args:
    y - vector of true labels (make sure they are 1 indexed)
    ŷ - matrix (n_inputs × outputs) of predicted logits
"""
function acc_fun(y::AbstractArray{<:Number}, ŷ::AbstractArray)
    count(round.(σ.(ŷ)) .== y) / length(y)
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
        ŷ = vcat(ŷ...)
        mae = mae_fun(y_test, ŷ)
        mse = mse_fun(y_test, ŷ)
        # need to add one as labels are 0 indexed
        acc = acc_fun(y_labels, ŷ)
        return nn_poly, mae, mse, acc
    else
        return nn_poly
    end   
end


# min and max used for MinMax scaler during training of the NN
data_min = -9 .* ones(23)
data_max = [93, 789, 383, 383, 74, 17, 16, 100, 83, 9, 8, 87, 17, 100, 24, 66, 66, 232, 471, 32, 23, 18, 100];

f_heloc = CSV.File(string(@__DIR__, "/data/heloc_dataset.csv"))
X_test = [Float64.([x for x in f_heloc[i]][2:end]) for i in 1:size(f_heloc, 1)]
X_test = [(x .- data_min) ./ (data_max .- data_min) for x in X_test]
y_test = [[x for x in f_heloc[i]][1] for i in 1:size(f_heloc, 1)]
y_test = [ifelse(y == "Good", 1., 0.) for y in y_test]

model_file = string(@__DIR__, "/networks/heloc.onnx")
net = VNNLib.load_network(model_file)

ŷ = [net(x) for x in X_test]
ŷ = vcat(ŷ...)

acc_fun(y_test, ŷ)


z = Zonotope(I(23) .* 0.5, zeros(23) .+ 0.5, I(23))

z_out = net(z, PropState(true));

println("===== HELOC Iterative =====")
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

jldsave(string("heloc_zono_polys_data_", now(), ".jld2"); nets, maes, mses, accs, times)


model_file = string(@__DIR__, "/networks/heloc_2e5.onnx")
net = VNNLib.load_network(model_file)

ŷ = [net(x) for x in X_test]
ŷ = vcat(ŷ...)

println("===== HELOC L1 Iterative =====")
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

jldsave(string("heloc_l1_zono_polys_data_", now(), ".jld2"); nets_l1, maes_l1, mses_l1, accs_l1, times_l1)



plot((1:9) ∪ (10:10:200),  maes, label="maes", marker=:diamond, xlabel="degree", ylabel="MAE", yaxis=:log, title="HELOC")
plot!((1:9) ∪ (10:10:200),  maes_l1, label="maes L1", marker=:diamond)

plot((1:9) ∪ (10:10:200),  mses, label="mses", marker=:diamond, xlabel="degree", ylabel="MSE", yaxis=:log, title="HELOC")
plot!((1:9) ∪ (10:10:200),  mses_l1, label="mses L1", marker=:diamond)

plot((1:9) ∪ (10:10:200),  accs, label="acc", marker=:diamond, xlabel="degree", ylabel="accuracy", title="HELOC", legend=:inside)
plot!((1:9) ∪ (10:10:200),  accs_l1, label="acc L1", marker=:diamond)

plot((1:9) ∪ (10:10:200),  times, label="t", marker=:diamond, xlabel="degree", ylabel="time (sec)", title="HELOC")
plot!((1:9) ∪ (10:10:200),  times_l1, label="t L1", marker=:diamond)






#################################
### Verified Equivalence      ###
#################################

degrees = (1:9) ∪ (10:10:200)

#res = load("./mnist_4x256_1e4_zono_polys_data_2025-03-27T17:39:57.751.jld2")
#nets_l1_1e4 = res["nets_l1_1e4"];
#maes_l1_1e4 = res["maes_l1_1e4"];

model_file = sring(@__DIR__, "/networks/heloc.onnx")
model_file = string(@__DIR__, "/networks/heloc_2e5.onnx")
nn = VNNLib.load_network(model_file)

VeryDiff.OPTIM_ITERS[] = 0
prop_state = PropState(true)
∂bounds_l1 = []
for (i, nn_poly) in enumerate(nets_l1)
    # propagate differential zonotope through the difference network
    nn_diff = GeminiNetwork(nn_poly, nn);

    # need to convert to matrix, s.t. z and ∂z have the same type
    z = Zonotope(Matrix(I(23)) .* 0.5, zeros(23) .+ 0.5, I(23))
    ∂z = Zonotope(zeros(23, 23), zeros(23), nothing)
    zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)
    #@profile ẑΔ = nn_diff(zΔ, PropState(true))
    ẑΔ = nn_diff(zΔ, PropState(true))

    bounds_diff = zono_bounds(ẑΔ.∂Z)
    ∂bound = maximum(abs.(bounds_diff))
    println("\ndegree = ", degrees[i], " ∂bound = ", ∂bound)
    println("\tlbs = ", bounds_diff[:,1])
    println("\tubs = ", bounds_diff[:,2])
    println("")
    push!(∂bounds_l1, ∂bound)
end

∂bounds_l1 = copy(∂bounds)


model_file = string(@__DIR__, "/networks/heloc.onnx")
nn = VNNLib.load_network(model_file)

VeryDiff.OPTIM_ITERS[] = 0
prop_state = PropState(true)
∂bounds = []
for (i, nn_poly) in enumerate(nets)
    # propagate differential zonotope through the difference network
    nn_diff = GeminiNetwork(nn_poly, nn);

    # need to convert to matrix, s.t. z and ∂z have the same type
    z = Zonotope(Matrix(I(23)) .* 0.5, zeros(23) .+ 0.5, I(23))
    ∂z = Zonotope(zeros(23, 23), zeros(23), nothing)
    zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)
    ẑΔ = nn_diff(zΔ, PropState(true))

    bounds_diff = zono_bounds(ẑΔ.∂Z)
    ∂bound = maximum(abs.(bounds_diff))
    println("\ndegree = ", degrees[i], " ∂bound = ", ∂bound)
    println("\tlbs = ", bounds_diff[:,1])
    println("\tubs = ", bounds_diff[:,2])
    println("")
    push!(∂bounds, ∂bound)
end


#plot((1:9) ∪ (10:10:200),  maes, label="maes_l1", marker=:diamond, xlabel="degree", ylabel="error", yaxis=:log, title="HELOC")
plot((1:9) ∪ (10:10:200),  ∂bounds, label="bounds", marker=:diamond, xlabel="degree", ylabel="error", yaxis=:log, title="HELOC Verified Difference")
plot!((1:9) ∪ (10:10:200),  ∂bounds_l1, label="bounds L1", marker=:diamond)
