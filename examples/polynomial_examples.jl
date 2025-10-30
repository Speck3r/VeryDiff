
using VNNLib, VeryDiff 

# load network using VNNLib.OnnxParser
model_path = joinpath(@__DIR__, "..", "test", "examples", "networks", "mnist_256x4_1e4.onnx")
model = load_onnx_model(model_path)
lmodel = VeryDiff.to_layered_model(model)

# compute polynomial approximation (usually much higher degree than 5 needed)
z = Zonotope(zeros(784), ones(784))
degree = 5
nn_poly = VeryDiff.approximate_polynomial_iterative(lmodel, z, degree, cheby=true, verbosity=1, max_iter=20)

# evaluate the networks on an example input
lmodel(zeros(784))

nn_poly(zeros(784))

# equivalence verification
# construct gemini network
nn_diff = GeminiNetwork(nn_poly, lmodel)

# make difference zonotope
∂z = Zonotope(zero(z.G), zero(z.c), nothing)
zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)

# extract approximation intervals for each polynomial
dom = VeryDiff.extract_approximation_domain(nn_poly)

# propagate difference through the networks
ẑΔ = nn_diff(zΔ, PropState(true), dom, nothing)

# concretize difference zonotope
bounds_diff = zono_bounds(ẑΔ.∂Z)