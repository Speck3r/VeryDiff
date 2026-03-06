module Transformers

using LinearAlgebra
using Optim

using VNNLib
using VNNLib.OnnxParser: ONNXLinear, ONNXRelu, ONNXAddConst, ONNXGelu

using TimerOutputs

import ..VeryDiff: zono_bounds, @simd_bool_expr
import ..VeryDiff: islinear, REMEZ_ITERS, OPTIM_ITERS, approx_polynomial_lin, clenshaw_chebyshev, ChebyshevPolynomial, make_eval_poly, dpoly, real_roots
using VeryDiff

using ..Definitions
using ..Debugger

include("Util.jl")
include("Init.jl")
include("Polynomial_Relaxations.jl")
include("PolynomialDifferenceRelaxation.jl")
include("Single_Transformers.jl")
include("Diff_Transformers.jl")
include("Network.jl")

export propagate!

end