
using VNNLib, VeryDiff, Plots
const OXP = VNNLib.OnnxParser

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false

# added the following lines of code to 
# - onnx2pytorch/convert/attribute.py:
#         elif attr.name == "training_mode":
#            training_mode = extract_attr_values(attr)
#            assert training_mode == 0, "Only inference mode supported, but got training_mode={}".format(training_mode)
#
# - onnx2pytorch/convert/layer.py:
#         layer.eval()

model_path = joinpath(@__DIR__, "..", "..", "VeryDiffPolyExperiments", "networks", "cifar", "cifar_conv2_ultra_tiny.onnx")
model = load_onnx_model(model_path)

model_poly = VeryDiff.approximate_polynomial_abcrown(model_path, 100, verbosity=1);

input_data = Dict(k => rand(v...) for (k, v) in model_poly.input_shapes)
model_poly_dense = VNNLib.net2dense(model_poly, input_data)

x = rand(32, 32, 3, 1);
y = compute_output(model, x)
y_poly = compute_output(model_poly_dense, vec(x));

abs.(y .- y_poly)

model_dense = VNNLib.net2dense(model, input_data);
verification_pass(model_poly_dense, model_dense, zeros(3*32*32), ones(3*32*32))