
using VeryDiff, LinearAlgebra, VNNLib, DoubleFloats, Plots
import VeryDiff: approximate_polynomial_iterative, approximate_polynomial


model_file = "./test/examples/networks/mnist_256x4_2e5.onnx"
poly_dir = "./test/examples/poly_coeffs/models/mnist_fc/poly_2/"

# need to make it Matrix(I(...)) because it has type Diagonal otherwise
z = Zonotope(Matrix(I(784)) .* 0.05, zeros(784), Float64.(I(784)));

nn = VNNLib.load_network(model_file)

test_set = [VeryDiff.random_point(z) for _ in 1:10000]
ys = [nn(x) for x in test_set]
Y  = hcat(ys...)';

nn_poly = approximate_polynomial_iterative(nn, z, 100, verbosity=1, cheby=true)

bnds_relu = VeryDiff.get_zono_bounds(nn, z)
bnds_poly = VeryDiff.get_zono_bounds(nn_poly, z);


using Profile

VeryDiff.OPTIM_ITERS[] = 0
prop_state = PropState(true)

ẑ_poly = nn_poly(z, prop_state)
bounds_poly = zono_bounds(ẑ_poly)

# propagate differential zonotope through the difference network
nn_diff = GeminiNetwork(nn_poly, nn);
∂z = Zonotope(zeros(784, 784), zeros(784), nothing)
zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)
#@profile ẑΔ = nn_diff(zΔ, PropState(true))
ẑΔ = nn_diff(zΔ, PropState(true))


bounds_diff = zono_bounds(ẑΔ.∂Z)


