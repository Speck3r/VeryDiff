
using VNNLib, VeryDiff, HDF5, Plots
const OXP = VNNLib.OnnxParser

# load network using VNNLib.OnnxParser
model_path = joinpath(@__DIR__, "..", "test", "examples", "networks", "mnist_256x4_1e4.onnx")
model = load_onnx_model(model_path)
lmodel = VeryDiff.to_layered_model(model)


pythonpath = "/home/philipp/miniconda3/envs/errornodes/bin/python"

##################################
# add error nodes to the network #
##################################
scriptpath = joinpath(@__DIR__, "..", "..", "ErrorNodes", "src", "insert_error_nodes.py")
error_net_path = "error_net.onnx"
run(`$pythonpath $scriptpath $(model_path) $(error_net_path)`)

##############################
# get bounds for first layer #
##############################
input_bounds_file = "input_bounds.h5"
h5open(input_bounds_file, "w") do file 
    for (k, v) in model.input_shapes
        lb = zeros(v)
        ub = ones(v)
        # hdf5 already converts from WHCN to NCHW
        file[k] = cat(lb, ub, dims=ndims(lb))
    end
end

scriptpath = joinpath(@__DIR__, "..", "..", "ErrorNodes", "src", "get_bounds.py")
error_net_path = "error_net.onnx"
output_name = lmodel.layers[2].inputs[1]
output_bounds = "out_bounds.h5"
run(`$pythonpath $scriptpath $(error_net_path) $(input_bounds_file) $(output_name) --output_file $(output_bounds)`)

h5open(output_bounds, "r") do file 
    bounds = read(file[output_name])
end


###################################
# get bounds for the second layer #
###################################
input_bounds_file = "input_bounds.h5"

activations_considered = [lmodel.layers[2]]
error_magnitudes = [0.05 .* ones(256, 1)]
h5open(input_bounds_file, "w") do file 
    for (k, v) in model.input_shapes
        lb = zeros(v)
        ub = ones(v)
        # hdf5 already converts from WHCN to NCHW
        file[k] = cat(lb, ub, dims=ndims(lb))
    end

    for (l, ϵs) in zip(activations_considered, error_magnitudes)
        error_name = "eps_" * l.name
        file[error_name] = ϵs
    end
end

scriptpath = joinpath(@__DIR__, "..", "..", "ErrorNodes", "src", "get_bounds.py")
error_net_path = "error_net.onnx"
output_name = lmodel.layers[4].inputs[1]
output_bounds = "out_bounds.h5"
run(`$pythonpath $scriptpath $(error_net_path) $(input_bounds_file) $(output_name) --output_file $(output_bounds)`)

bounds = h5open(output_bounds, "r") do file 
    read(file[output_name])
end



## all in a single method
const ABCROWN_PYTHONPATH = "/home/philipp/miniconda3/envs/errornodes/bin/python"
const ERROR_NODES_SCRIPT = joinpath(@__DIR__, "..", "..", "ErrorNodes", "src", "insert_error_nodes.py")
const BOUNDS_SCRIPT      = joinpath(@__DIR__, "..", "..", "ErrorNodes", "src", "get_bounds.py")

function compute_approximation_errors(bounds; ϵ=0.05)
    lbs = bounds[:,1:1]
    ubs = bounds[:,2:2]

    # TODO: return real polynomial approximaion errors
    return ϵ .* ones(size(lbs))
end

function approximate_polynomial_abcrown(onnx_path, degree; cheby=true, verbosity=0, max_iter=20, max_polys_per_layer=Inf)
    # load model in julia
    model = load_onnx_model(onnx_path)
    lmodel = VeryDiff.to_layered_model(model)

    # extend network by error inputs
    error_net_path = "error_net.onnx"
    run(`$(ABCROWN_PYTHONPATH) $(ERROR_NODES_SCRIPT) $(onnx_path) $(error_net_path)`)

    input_bounds_file = "input_bounds.h5"
    activations_considered = []
    error_magnitudes = []
    layer_bounds = []
    layers_poly = Vector{OXP.Node}()

    for (i, l) in enumerate(lmodel.layers)
        if (l isa OXP.ONNXRelu) || (l isa OXP.ONNXGelu)
            # prepare input bounds
            h5open(input_bounds_file, "w") do file 
                for (k, v) in model.input_shapes
                    lb = zeros(v)
                    ub = ones(v)
                    # hdf5 already converts from WHCN to NCHW
                    file[k] = cat(lb, ub, dims=ndims(lb))
                end

                for (l, ϵs) in zip(activations_considered, error_magnitudes)
                    error_name = "eps_" * l.name
                    file[error_name] = reshape(ϵs, :, 1)  # append a batch dimension
                end
            end

            output_name = l.inputs[1]
            output_bounds = "out_bounds.h5"
            run(`$ABCROWN_PYTHONPATH $(BOUNDS_SCRIPT) $(error_net_path) $(input_bounds_file) $(output_name) --output_file $(output_bounds)`)

            bounds = h5open(output_bounds, "r") do file 
                read(file[output_name])
            end

            # ϵs = compute_approximation_errors(bounds, ϵ=0.05)
            layer_poly, ϵs = VeryDiff.approximate_polynomial(l, bounds, degree, cheby=cheby, verbosity=verbosity, max_iter=max_iter, max_polys_per_layer=max_polys_per_layer)

            push!(activations_considered, l)
            push!(error_magnitudes, ϵs)
            push!(layer_bounds, bounds)

            verbosity > 0 && println("--- layer $i ---")
            verbosity > 0 && println("lower = ", bounds[:,1][1:min(size(bounds, 1), 5)])
            verbosity > 0 && println("upper = ", bounds[:,2][1:min(size(bounds, 1), 5)])
            !all(isfinite.(bounds)) && println("lb non-finite: ", (1:size(bounds,1))[.~isfinite.(bounds[:,1])])
            !all(isfinite.(bounds)) && println("ub non-finite: ", (1:size(bounds,1))[.~isfinite.(bounds[:,2])])
        else
            layer_poly = l
        end
        push!(layers_poly, layer_poly)
    end

    # need to do [l for l in layers_poly] to convert from OXP.Node without information about identifier type to OXP.Node{S}
    return VeryDiff.LayeredModel([l for l in layers_poly]), activations_considered, error_magnitudes, layer_bounds
end

using VeryDiff
model_path = joinpath(@__DIR__, "..", "test", "examples", "networks", "mnist_256x4_1e4.onnx")
degree = 5
# model_poly, activations_considered, error_magnitudes, layer_bounds = VeryDiff.approximate_polynomial_abcrown(model_path, degree, verbosity=1);
model_poly = VeryDiff.approximate_polynomial_abcrown(model_path, degree, verbosity=1);


degree = 10
model_poly2, activations_considered2, error_magnitudes2, layer_bounds2 = approximate_polynomial_abcrown(model_path, degree, verbosity=1);


lbs = vcat([bound[:,1] for bound in layer_bounds]...)
ubs = vcat([bound[:,2] for bound in layer_bounds]...)

lbs2 = vcat([bound[:,1] for bound in layer_bounds2]...)
ubs2 = vcat([bound[:,2] for bound in layer_bounds2]...)


plot(lbs, label="lower bounds")
plot!(ubs, label="upper bounds")
plot!(lbs2, label="lower bounds 2", alpha=0.5)
plot!(ubs2, label="upper bounds 2", alpha=0.5)


##### Better solution: Just put abCROWN in the artifact, put the two python scripts in the source folder 