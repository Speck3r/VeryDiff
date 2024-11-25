
using VNNLib, TimerOutputs, LinearAlgebra
import CSV

module Dev
using VNNLib, VeryDiff, TimerOutputs, LinearAlgebra

const to = VeryDiff.to
const NEW_HEURISTIC = true

"""
Test is polynomial represented by coefficients [p₀, p₁, ...]
represents a linear function.
"""
islinear(v) = length(v) < 3 ? true : all(v[3:end] .== 0)

struct Poly{N} <: VNNLib.Layer where {N<:Number}
    coeffs::Array{N}
end

function (L::Poly)(x::Vector{N}) where {N<:Number}
    n_neurons, n_coeffs = size(L.coeffs)
    degree = n_coeffs - 1
    vec(sum(L.coeffs .* x .^ collect(0:degree)', dims=2))
end

function (L::Poly)(Z::Zonotope, P::PropState; bounds=nothing)
    # TODO: what about polynomials that are actually linear functions?
    #       Treat them separately?
    return @timeit to "Zonotope_PolyProp" begin
        @timeit to "Bounds" begin
            row_count = size(Z.G, 1)
            if isnothing(bounds)
                bounds = zono_bounds(Z)
            end
            lower = @view bounds[:, 1]
            upper = @view bounds[:, 2]
        end

        @timeit to "Vectors" begin
            nonlinmask = .~islinear.(eachrow(L.coeffs))
            λ = copy(L.coeffs[:,2])
            β = copy(L.coeffs[:,1])
            γ = zeros(row_count)

            # TODO is there a better way than eachrow()?
            res = VeryDiff.approx_polynomial_lin.(eachrow(L.coeffs[nonlinmask, :]), lower[nonlinmask], upper[nonlinmask])
            λ[nonlinmask] .= getindex.(res, 1)  # slope of the input
            β[nonlinmask] .= getindex.(res, 2)  # bias 
            γ[nonlinmask] .= getindex.(res, 3)  # new error

            ĉ = λ .* Z.c .+ β
        end

        @timeit to "Influence Matrix" begin
            if NEW_HEURISTIC
                # TODO(steuber): Can we avoid this reallocation?
                @timeit to "Allocation" begin
                    influence_new = zeros(Float64, size(Z.influence, 1), size(Z.influence, 2) + row_count)
                end
                @timeit to "Set Matrix" begin
                    influence_new[:, 1:size(Z.influence, 2)] .= Z.influence
                end
                @timeit to "Multiply" begin
                    influence_new[:, (size(Z.influence, 2)+1):end] .= abs.(Z.influence) * abs.(Z.G)'
                end
            else
                influence_new = Z.influence
            end
        end

        @timeit to "Allocation" begin
            Ĝ = zeros(Float64, row_count, size(Z.G, 2) + row_count)
        end
        #Z.G .*= λ
        @timeit to "Set Matrix" begin
            Ĝ[:, 1:size(Z.G, 2)] .= Z.G
            Ĝ[:, size(Z.G, 2)+1:end] .= I(row_count)
        end
        @timeit to "Column Multiply" begin
            Ĝ[:, 1:size(Z.G, 2)] .*= λ
            Ĝ[:, size(Z.G, 2)+1:end] .*= abs.(γ)
        end

        return Zonotope(Ĝ, ĉ, influence_new)
    end
end
end


function read_poly_coeffs(model_dir, poly_dir)
    nn = VNNLib.load_network(model_dir)
    activation_layers = sum(typeof.(nn.layers) .== VNNLib.ReLU)

    coeffs = []
    for layer in 0:activation_layers-1
        f = CSV.File(string(poly_dir, "/coeffs_", layer, ".csv"), header=false)
        n_neurons = length(f)
        n_coeffs = n_neurons > 0 ? length(f[1]) : 0

        coeffs_layer = zeros(n_neurons, n_coeffs)
        for (i, line) in enumerate(f)
            for j in 1:length(line)
                coeffs_layer[i, j] = line[j]
            end
        end

        push!(coeffs, coeffs_layer)
    end

    return coeffs
end


function load_polynomial_nn(model_dir, poly_dir)
    nn = VNNLib.load_network(model_dir)
    coeffs = read_poly_coeffs(model_dir, poly_dir)

    layers_new = Vector{VNNLib.Layer}()
    cnt = 1
    for L in nn.layers
        if typeof(L) == VNNLib.ReLU
            player = Dev.Poly(coeffs[cnt])
            cnt += 1

            push!(layers_new, player)
        else
            # TODO: does this cause problems because it's no copy?
            push!(layers_new, L)
        end
    end

    return VNNLib.Network(layers_new)
end

model_dir = "../vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x6.onnx"
poly_dir  = "../PolynomialEquivalenceNN/src/poly_coeffs/models/mnist_fc/poly_2"

nn = VNNLib.load_network("../vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x6.onnx")
f = CSV.File(poly_dir * "/coeffs_0.csv", header=false)

nn_poly = load_polynomial_nn(model_dir, poly_dir);

z = VeryDiff.Zonotope(I(784) .* 0.05, zeros(784), I(784));

# what is the PropState?
ẑ = nn_poly(z, PropState(true))
