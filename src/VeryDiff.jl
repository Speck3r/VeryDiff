module VeryDiff

using Printf

using DoubleFloats  # for better precision in polynomial approximation
#using MaskedArrays
using LinearAlgebra
#using SparseArrays
using VNNLib
#using ThreadPinning
using CSV
using PolynomialRoots
using Optim
using JLD2
using Artifacts
using PythonCall
using HDF5
using ThreadsX

using GLPK
using Gurobi

const OXP = VNNLib.OnnxParser

const USE_GUROBI = Ref{Bool}(true)
const NEW_HEURISTIC = Ref{Bool}(true)
const USE_DIFFZONO = Ref{Bool}(true)

    
"""use p(x) - ReLU(y) = p(x) - (x - Δ) in the active ReLU case instead of p(x) - y"""
const USE_REWRITE_DIFF = Ref{Bool}(true)

"""use bounds from difference zonotope to tighten individual network bounds"""
const TIGHTEN_BOUNDS_DIFF = Ref{Bool}(false)

"""if imaginary part is smaller than IMAG_TOL, we count it as a real number"""
const IMAG_TOL = Ref{Float64}(1e-10)

"""number of Remez algorithm iterations for polynomial approximation"""
const REMEZ_ITERS = Ref{Int}(10)

"""number of iterations for finding good relaxation for p(x)-ReLU(y)"""
const OPTIM_ITERS = Ref{Int}(0)

"""print warning if leading coefficient of polynomial for which we want to compute roots is close to zero"""
const ALMOST_ZERO_LEADING_COEFF_WARNING = Ref{Bool}(true)

"""if leading coefficient of chebyshev polynomial has abs value smaller than this, we assume it is zero for roots computation"""
const ROOTS_ALMOST_ZERO_TOL = Ref{Float64}(1e-15)

const ABCROWN_PATH = Ref{String}("")
const AUTOLIRPA_PATH = Ref{String}("")
const PYTHON_SCRIPTS_DIR = joinpath(@__DIR__, "..", "python")

"""number of threads to use for parallel polynomial approximation of layers"""
const APPROX_POLY_THREADS = Ref{Int}(1)

# We have our own multithreadding so we don't want to use BLAS multithreadding
function __init__()
    BLAS.set_num_threads(1)
    try
        # this might not be necessary, if USE_GUROBI is settable
        if USE_GUROBI[]
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
    catch e 
        println("Gurobi error: ", e)
        println("!!! falling back to GLPK !!!")
        USE_GUROBI[] = false
    end

    # artifacts only downloads alpha-beta-CROWN without the auto_LiRPA submodule, so we need to add both as separate artifacts.
    # TODO: can we move auto_LiRPA into its place in alpha-beta-CROWN automatically?
    ABCROWN_PATH[] = artifact"alpha-beta-CROWN"
    AUTOLIRPA_PATH[] = artifact"auto_LiRPA"

    # want to find first valid version of abCROWN or auto_LiRPA, even if we downloaded multiple ones.
    # avoid using the hard-coded commit names in the statements below
    get_inner(root) = joinpath(root, first(filter(x -> isdir(joinpath(root, x)), readdir(root))))
    ABCROWN_PATH[] = get_inner(ABCROWN_PATH[])
    AUTOLIRPA_PATH[] = get_inner(AUTOLIRPA_PATH[])
    pyimport("sys")."path".append(ABCROWN_PATH[])
    pyimport("sys")."path".append(AUTOLIRPA_PATH[])

    # don't hard-code the commit names (here main and master)
    #pyimport("sys")."path".append(joinpath(ABCROWN_PATH[], "alpha-beta-CROWN-main"))
    #pyimport("sys")."path".append(joinpath(AUTOLIRPA_PATH[], "auto_LiRPA-master"))
    pyimport("sys")."path".append(PYTHON_SCRIPTS_DIR)
end

#pinthreads(:cores)

const FIRST_ROUND = Ref{Bool}(true)

include("Util/simd_bool.jl")
include("Debugger/Debugger.jl")

# new ONNX nodes for polynomials in Definitions.jl need chebyshev info
include("polynomials/Util.jl")
include("polynomials/chebyshev/Chebyshev.jl")
include("polynomials/chebyshev/chebyshev_interface.jl")
include("polynomials/PiecewisePolynomials.jl")

# verified error of 1e-10
const GELU_PP = load_piecewise_poly(joinpath(@__DIR__, "..", "resources", "gelu_6pieces_15degree_1e-10error.jld2"))
# same parameters - *sampled* error of 2.48e-14 (not verified)
# const GELU_PP = load_piecewise_poly(joinpath(@__DIR__, "..", "resources", "gelu_piecewise_poly_sampled_degree_15.jld2"))

include("polynomials/Remez.jl")
# include("polynomials/PolynomialDifferenceRelaxation.jl")

include("Definitions/Definitions.jl")
using .Definitions

include("Transformers/Transformers.jl")
using .Transformers

using JuMP

const GRB_ENV = Ref{Any}(nothing)

include("polynomials/ApproximateNetworkPoly.jl")

include("MultiThreadding.jl")

include("Properties/Properties.jl")
using .Properties

include("Verifier.jl")


# command line interface
include("Cli.jl")

export Network,GeminiNetwork,Layer,Dense,ReLU,WrappedReLU
export Zonotope, DiffZonotope, PropState
export zono_optimize, zono_bounds
export verify_network, verification_pass
export get_epsilon_property, epsilon_split_heuristic, get_epsilon_property_naive
export get_top1_property, top1_configure_split_heuristic

end
