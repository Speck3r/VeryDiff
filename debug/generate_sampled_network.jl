
using VNNLib, VeryDiff 

model_path = joinpath(@__DIR__, "..", "test", "examples", "networks", "mnist_256x4_1e4.onnx")
model = load_onnx_model(model_path)
X = [reshape(zeros(784), 28, 28, 1, 1), reshape(ones(784), 28, 28, 1, 1)];

degree = 5
nn_sampled = VeryDiff.approximate_polynomial_iterative_sampling(model, X, degree, cheby=true, verbosity=1, max_polys_per_layer=1)


model_path = joinpath(@__DIR__, "..", "..", "VeryDiffPolyExperiments", "networks", "cifar", "best_model_bn_8_0.0005l1.onnx")
model = load_onnx_model(model_path);
X = [reshape(rand(32*32*3), 32, 32, 3, 1) for i in 1:10];

degree = 5
nn_sampled = VeryDiff.approximate_polynomial_iterative_sampling(model, X, degree, cheby=true, verbosity=1, max_polys_per_layer=Inf)