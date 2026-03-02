
using VNNLib, VeryDiff, Plots
const OXP = VNNLib.OnnxParser


# Numerical error is solved for now: 
# The problem was that an interval intersection was computed incorrectly and we computed the value of a polynomial way outside of its approximation domain, which led to huge errors.
# We may need to come back here if there are still numerical issues.

model_path = joinpath(@__DIR__, "..", "..", "VeryDiffPolyExperiments", "networks", "mnist", "mnist_gelu_256x4_1e4.onnx")
model = load_onnx_model(model_path)
lmodel = VeryDiff.to_layered_model(model)

model_poly = VeryDiff.approximate_polynomial_abcrown(model_path, 20, tight_gelu=false)

model_poly = VeryDiff.approximate_polynomial_abcrown(model_path, 95, tight_gelu=false)

#l = -7.615006613076782
#u = -6.092005290461426
l, u = -0.005752564919215014, 0.005532740165177755

p, ϵ = VeryDiff.approx_gelu_poly(l, u, 95, verbosity=1, max_iter=20)

C = VeryDiff.colleague_matrix(p)

gpp = VeryDiff.GELU_PP
