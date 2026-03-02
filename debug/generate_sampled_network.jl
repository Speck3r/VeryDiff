
using VNNLib, VeryDiff 

# load network using VNNLib.OnnxParser
model_path = joinpath(@__DIR__, "..", "test", "examples", "networks", "mnist_256x4_1e4.onnx")
model = load_onnx_model(model_path)
lmodel = VeryDiff.to_layered_model(model)

X = [zeros(784), ones(784)]

# generate sampled network
degree = 5
nn_sampled = VeryDiff.approximate_polynomial_iterative_sampling(lmodel, X, degree, cheby=true, verbosity=1, max_polys_per_layer=1)