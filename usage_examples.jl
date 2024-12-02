
using VeryDiff, LinearAlgebra, VNNLib, Optim
import VeryDiff: load_polynomial_nn, load_approximation_bounds

model_file = "../vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x6.onnx"
poly_dir  = "../PolynomialEquivalenceNN/src/poly_coeffs/models/mnist_fc/poly_2"

nn = VNNLib.load_network(model_file)
nn_poly = load_polynomial_nn(model_file, poly_dir);
nn_diff = GeminiNetwork(nn_poly, nn);

bounds = load_approximation_bounds(nn, poly_dir)

# propagate zonotopes through the individual networks
# just some radius around 0
z = VeryDiff.Zonotope(I(784) .* 0.05, zeros(784), I(784));
ẑ = nn(z, PropState(true))
ẑ_poly = nn_poly(z, PropState(true))
ẑ_poly_bnds = nn_poly(z, PropState(true), bounds)

# propagate differential zonotope through the difference network
∂z = VeryDiff.Zonotope(zeros(784, 784), zeros(784), nothing)
# differential zonotope is consists of zono for the first net, zono for the second net and zono for the difference
zΔ = VeryDiff.DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)

ẑΔ = nn_diff(zΔ, PropState(true))


# find a good estimate on the lower bound of the difference of the nns 
optfun = x -> begin
    y = nn(x)
    y_poly = nn_poly(x)
    # Optim minimizes by default, while we want max
    -maximum(abs.(y .- y_poly))
end

x₀ = zeros(784)
res = optimize(optfun, -0.05 .* ones(784), 0.05 .* ones(784), x₀, SAMIN(), 
               Optim.Options(show_trace=true, show_every=1000, iterations=10000))







