module VeryDiff

using Printf

#using MaskedArrays
using LinearAlgebra
#using SparseArrays
using VNNLib
#using ThreadPinning
using CSV
using PolynomialRoots
using Optim

using GLPK

NEW_HEURISTIC = true
USE_GUROBI = true

USE_DIFFZONO = true

"""use p(x) - ReLU(y) = p(x) - (x - Δ) in the active ReLU case instead of p(x) - y"""
const USE_REWRITE_DIFF = Ref{Bool}(false)

"""use bounds from difference zonotope to tighten individual network bounds"""
const TIGHTEN_BOUNDS_DIFF = Ref{Bool}(false)

"""if imaginary part is smaller than IMAG_TOL, we count it as a real number"""
const IMAG_TOL = Ref{Float64}(1e-10)

"""number of Remez algorithm iterations for polynomial approximation"""
const REMEZ_ITERS = Ref{Int}(1)

# We have our own multithreadding so we don't want to use BLAS multithreadding
function __init__()
    BLAS.set_num_threads(1)
    GRB_ENV[] = Gurobi.Env()
    GRBsetintparam(GRB_ENV[], "OutputFlag", 0)
    GRBsetintparam(GRB_ENV[], "LogToConsole", 0)
    GRBsetintparam(GRB_ENV[], "Threads", 0)
    #GRBsetintparam(GRB_ENV[], "Method", 2)
    #       mnist_19_local_21.vnnlib        mnist_18_local_18
    #0 :    0.018826400587219343s/loop      0.03304489948205128s/loop
    #1 :    0.01705984154058722s/loop       0.03352098044717949s/loop
    #2 :    0.020955224224525042s/loop      0.038390683782564106s/loop
end

#pinthreads(:cores)

FIRST_ROUND = true

using TimerOutputs
const to = TimerOutput()

using JuMP
#using GLPK
using Gurobi

const GRB_ENV = Ref{Any}(nothing)

include("Debugger.jl")
include("Chebyshev.jl")
include("Definitions.jl")
include("Network.jl")
include("Util.jl")
include("Zonotope.jl")
include("Remez.jl")
include("PolynomialDifferenceRelaxation.jl")
include("Layers_Zonotope.jl")
include("Layers_DiffZonotope.jl")
include("MultiThreadding.jl")
include("Properties.jl")
include("Verifier.jl")
include("ApproximateNetworkPoly.jl")

export Network,GeminiNetwork,Layer,Dense,ReLU,WrappedReLU
export parse_network
export Zonotope, DiffZonotope, PropState
export zono_optimize, zono_bounds
export verify_network
export get_epsilon_property, epsilon_split_heuristic, get_epsilon_property_naive
export get_top1_property, top1_configure_split_heuristic

end # module AlphaZono
