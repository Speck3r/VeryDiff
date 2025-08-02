using Test

using VeryDiff
using VNNLib
using VNNLib.OnnxParser

const OXP = VNNLib.OnnxParser

function create_random_inputs(model::OXP.OnnxNet)
    input_data = Dict{String, AbstractArray}()

    for (k, v) in model.input_shapes
        # Create random input data with the specified shape
        input_data[k] = rand(v...)
    end

    return input_data   
end

function test_matching_execution()
    # Run all onnx files in examples/nets with
    # - OXP.compute_outputs(model, input_data)
    # - network representation of VeryDiff (obtained via preprocess_onnx_model)
    # compare results on random input data
    blacklist = [
        "examples/nets/2_80-1-0.1.onnx",
        "examples/nets/2_80-1.onnx"
    ]
    onnx_files = readdir("examples/nets", join=true)
    for onnx_file in onnx_files
        # Ignoring some files due to softmax:
        # Softmax is handled implicitly in VeryDiff, so we skip parsing it
        if onnx_file in blacklist
            continue
        end
        if endswith(onnx_file, ".onnx")
            @info "Testing ONNX file: $onnx_file"
            # Load and preprocess the ONNX model
            net = OXP.load_onnx_model(onnx_file)
            preprocessed_net = VeryDiff.preprocess_onnx_model(net)
            input_data = create_random_inputs(net)
            # Compute outputs using OXP
            oxp_outputs = OXP.compute_outputs(net, input_data)
            # Extract output values
            oxp_output_values = [oxp_outputs[k] for k in keys(oxp_outputs)]
            # Compute outputs using VeryDiff's network representation
            inputs = ([input_data[k] for k in keys(input_data)])[1]
            verydiff_outputs = [preprocessed_net(reduce(vcat, inputs))]
            # Compare outputs
            for (oxp_output, verydiff_output) in zip(oxp_output_values, verydiff_outputs)
                @test isapprox(oxp_output, verydiff_output; atol=1e-6, rtol=1e-6)
            end
        end
    end
end

@testset "Onnx Execution" verbose=true begin
    test_matching_execution()
end