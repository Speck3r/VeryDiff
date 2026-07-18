module Transformers

using LinearAlgebra

using VNNLib
using VNNLib.OnnxParser: ONNXLinear, ONNXRelu, ONNXAddConst, ONNXLeakyRelu, ONNXSigmoid

using TimerOutputs

import ..VeryDiff: zono_bounds, @simd_bool_expr
using VeryDiff

using ..Definitions
using ..Debugger

const CUTOFF_SIGMOID_SLOPE = 1e-4 # used as a cutoff if abs(slope) smaller then set to 0, otherwise numerical problems
const LOOP_ITERATIONS_DIFF_SIGMOID = 10 # determine the number of iterations to search for a tangent point
const LOOP_ITERATIONS_SINGLE_SIGMOID = 10

include("Util.jl")
include("Init.jl")
include("Single_Transformers.jl")
include("Diff_Transformers.jl")
include("Network.jl")
include("Sigmoid/SigmoidDifferenceRelaxation.jl")

export propagate!

end