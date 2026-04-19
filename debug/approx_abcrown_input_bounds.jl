
# Just test that providing input bounds to abcrown approximation works.

using VNNLib, VeryDiff, Plots
const OXP = VNNLib.OnnxParser

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false

# model_path = joinpath(@__DIR__, "..", "..", "VeryDiffPolyExperiments", "networks", "mnist", "mnist_gelu_256x4_1e4.onnx")
model_path = joinpath(@__DIR__, "..", "..", "VeryDiffPolyExperiments", "networks", "cifar", "best_model_bn_8_0.0005l1.onnx")
model = load_onnx_model(model_path)

input_bounds_wide = Dict("input" => (zeros((32, 32, 3, 1)), ones((32, 32, 3, 1))))
input_bounds_tight = Dict("input" => (0.45 .* ones((32, 32, 3, 1)), 0.55 .* ones((32, 32, 3, 1))));

model_loose = VeryDiff.approximate_polynomial_abcrown(model_path, 20, tight_gelu=true)
model_tight = VeryDiff.approximate_polynomial_abcrown(model_path, 20, input_bounds=input_bounds_tight, tight_gelu=true)

n_loose_layers, io_map = VeryDiff.Definitions.sort_network(model_loose)
n_tight_layers, io_map = VeryDiff.Definitions.sort_network(model_tight)

ws_loose = vcat([L.node.u - L.node.l for L in n_loose_layers[[3,5,7,10]]]...)
ws_tight = vcat([L.node.u - L.node.l for L in n_tight_layers[[3,5,7,10]]]...)
p = plot(title="approximation bounds")
plot!(ws_loose, label="bounds wide")
plot!(ws_tight, label="bounds tight")
