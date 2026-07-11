using VeryDiff

sysimage_dir = @__DIR__


VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-0.1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_20-1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-0.1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_40-1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-0.1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-0.1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid-0.2.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid-0.3.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid-0.4.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])

VeryDiff.run_cmd([
    "--epsilon", "1000000",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid.onnx",
    "$sysimage_dir/../../test/examples/nets/4_20-1_Sigmoid-0.5.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.5.vnnlib"
])