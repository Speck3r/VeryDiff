
# Just test that providing input bounds to abcrown approximation works.

using VNNLib, VeryDiff, Plots
const OXP = VNNLib.OnnxParser

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false

model_path = joinpath(@__DIR__, "..", "..", "VeryDiffPolyExperiments", "networks", "collins", "NN_rul_small_window_20_gelu_1e-3l1_kernel_size.onnx")
model = load_onnx_model(model_path)

lbs = [-0.43787118, -0.4170134, -0.56220525, -0.47313478, -0.87071895, -1.71500653,
        -0.97503717, -2.93923836, 2.1, 0., -1.44232638, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
ubs = [5.12792935, 5.65973052, 5.13137438, 5.48585691, 4.33442824, 3.24138691,
        3.06227229, 1.77278949, 9.2, 1., 2.95009876, 1., 1., 1., 1., 1., 1., 1., 1., 1.]
lb = repeat(lbs, 1, 20, 1, 1)
ub = repeat(ubs, 1, 20, 1, 1)
input_bounds = Dict("input" => (lb, ub));

model_poly = VeryDiff.approximate_polynomial_abcrown(model_path, 60, input_bounds=input_bounds, tight_gelu=true)
