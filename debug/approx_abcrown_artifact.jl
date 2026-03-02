
using VNNLib, VeryDiff, Plots
const OXP = VNNLib.OnnxParser

model_path = joinpath(@__DIR__, "..", "..", "VeryDiffPolyExperiments", "networks", "mnist", "mnist_gelu_256x4_1e4.onnx")
model = load_onnx_model(model_path)
lmodel = VeryDiff.to_layered_model(model)

model_loose = VeryDiff.approximate_polynomial_abcrown(model_path, 20, tight_gelu=false)
model_tight = VeryDiff.approximate_polynomial_abcrown(model_path, 20, tight_gelu=true)

ws_loose = vcat([L.u - L.l for L in model_loose.layers[2:2:end]]...)
ws_tight = vcat([L.u - L.l for L in model_tight.layers[2:2:end]]...)
p = plot(title="approximation bounds")
plot!(ws_loose, label="aCROWN bounds")
plot!(ws_tight, label="tighter relax")

plot(ws_tight ./ ws_loose, title="bound improvement", label="w_tight / w_aCROWN")

plot(sort(ws_tight ./ ws_loose), title="bound improvement sorted", label="w_tight / w_aCROWN")
