
using VNNLib, VeryDiff 

model_path = joinpath(@__DIR__, "..", "test", "examples", "networks", "mnist_256x4_1e4")
model = load_onnx_model(model_path)
lmodel = VeryDiff.to_layered_model(model)

z = Zonotope(zeros(784), ones(784))
degree = 5
nn_poly = VeryDiff.approximate_polynomial_iterative(lmodel, z, degree, cheby=true, verbosity=1, max_iter=20)