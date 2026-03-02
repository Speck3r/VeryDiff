
using VeryDiff, VNNLib

nn1 = load_network(joinpath(@__DIR__, "..", "..", "verydiff-experiments", "benchmarks", "mnist-prune", "nets", "mnist_relu_3_100.onnx"));
nn2 = load_network(joinpath(@__DIR__, "..", "..", "verydiff-experiments", "benchmarks", "mnist-prune", "nets_pruned", "mnist_relu_3_100_0.79_adam_1_stepsize.onnx"));

f, n_in, _ = get_ast(joinpath(@__DIR__, "..", "..", "verydiff-experiments", "benchmarks", "mnist-prune", "specs", "mnist_90_global_4.vnnlib"));

for (bounds, _, _, num) in f
	prop = get_epsilon_property(1.)
	verify_network(nn1, nn2, bounds[1:n_in,:], prop, epsilon_split_heuristic, timeout=120)
end


