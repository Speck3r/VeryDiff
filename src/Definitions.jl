mutable struct Zonotope{N<:Number,GN<:AbstractMatrix{N},CN<:AbstractVector{N}}
    G::GN
    c::CN
    influence::Union{GN,Nothing}
end

function Zonotope(G::GN,c::CN,influence) where {GN,CN}
    Zonotope(G, c, isnothing(influence) ? nothing : GN(influence))
end


"""Construct a zonotope from lower and upper bounds."""
function Zonotope(low::AbstractVector{N}, high::AbstractVector{N}) where N<:Number
    @assert length(low) == length(high)
    mid = (low .+ high) ./ 2
    distance = (high .- low) ./ 2
    input_dim = length(low)
    G = distance .* Matrix(I, input_dim, input_dim)[:, distance .> 0]
    influence = Matrix(1.0I, sum(distance .> 0), sum(distance .> 0))
    return Zonotope(G, mid, influence)
end


"""
A verification task consists of an input specification and ...

args:
    middle - center of the input zonotope
    distance - radius of the input zonotope (only non-zero dimensions)
    distance_indices - indices of the non-zero dimensions
    ∂Z - zonotope representing the distance between the two networks
    verification_status - status of the verification task
    distance_bound - ??? (why is this initialized to 1.0?)
"""
struct VerificationTask{N<:Number,GN<:AbstractMatrix{N},CN<:AbstractVector{N}}
    middle :: Vector{N}
    distance :: Vector{N}
    distance_indices :: Vector{Int}
    ∂Z::Zonotope{N,GN,CN}
    verification_status
    distance_bound :: N
end

# Z₂ = Z₁ - ∂Z

mutable struct DiffZonotope{N<:Number,GN<:AbstractMatrix{N},CN<:AbstractVector{N}}
    Z₁::Zonotope{N,GN,CN}
    Z₂::Zonotope{N,GN,CN}
    ∂Z::Zonotope{N,GN,CN}
    num_approx₁ :: Int
    num_approx₂ :: Int
    ∂num_approx :: Int
end

mutable struct PropState
    first :: Bool
    i :: Int64
    num_relus :: Int64
    relu_config :: Vector{Int64}
    function PropState(first :: Bool)
        return new(first, 0, 0, Int64[])
    end
end

struct PropConfig end