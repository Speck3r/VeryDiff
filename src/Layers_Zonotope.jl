import VNNLib.NNLoader.Network
import VNNLib.NNLoader.Dense
import VNNLib.NNLoader.ReLU

function forward(N::Network, Z :: Zonotope, P :: PropState) :: Zonotope
    Zout = Z
    for L in N.layers
        Zout = forward(L, Zout, P)
    end
    return Zout
end

function _forward(::Tuple{}, Z, P) 
    return Z
end

function _forward(layers::Tuple{Dense,Vararg{<:Union{Dense,ReLU}}}, Z, P)
    Z_new = forward(first(layers), Z, P)
    return _forward(Base.tail(layers), Z_new, P)
end
function _forward(layers::Tuple{ReLU,Vararg{<:Union{Dense,ReLU}}}, Z, P)
    Z_new = forward(first(layers), Z, P)
    return _forward(Base.tail(layers), Z_new, P)
end

function forward(N::VeryDiffNetwork, Z :: Zonotope, P :: PropState) :: Zonotope
    return _forward(N.layers, Z, P)
end

function forward(L::Dense, Z :: Zonotope,P :: PropState) :: Zonotope
    return @timeit to "Zonotope_DenseProp" begin
    G = L.W * Z.G
    c = L.W * Z.c .+ L.b
    return Zonotope(G,c, Z.influence)
    end
end

function get_slope(l,u, alpha)
    if u <= 0
        return 0.0
    elseif l >= 0
        return 1.0
    else
        return alpha
    end
end

function influence_forward_relu(influence :: Nothing, Z, crossing)
    return nothing
end

function influence_forward_relu(influence :: Matrix{Float64}, Z, crossing)
    @timeit to "Allocation" begin
    influence_new = zeros(Float64, size(influence,1), size(influence,2)+count(crossing))
    end
    @timeit to "Set Matrix" begin
    influence_new[:,1:size(influence,2)] .= influence
    end
    @timeit to "Multiply" begin
    influence_new[:,(size(influence,2)+1):end] .=  abs.(influence) * abs.(@view Z.G[crossing,:])'
    end
    return influence_new
end

function forward(L::ReLU, Z :: Zonotope, P :: PropState; bounds = nothing) :: Zonotope
    return @timeit to "Zonotope_ReLUProp" begin
    @timeit to "Bounds" begin
    row_count = size(Z.G,1)
    if isnothing(bounds)
        bounds = zono_bounds(Z)
    end
    lower = @view bounds[:,1]
    upper = @view bounds[:,2]
    end

    @timeit to "Vectors" begin
    α = clamp.(upper./(upper.-lower),0.0,1.0)
    # Use is_onesided to compute 
    λ = ifelse.(upper.<=0.0,0.0,ifelse.(lower.>=0.0,1.0,α))

    crossing = lower.<0.0 .&& upper.>0.0
    
    γ = 0.5 .* max.(-λ .* lower,0.0,((-).(1.0,λ)).*upper)  # Computed offset (-λl/2)

    ĉ = λ .* Z.c .+ crossing.*γ
    end

    influence_new = influence_forward_relu(Z.influence, Z, crossing)

    @timeit to "Allocation" begin
    Ĝ = zeros(Float64,row_count, size(Z.G,2)+count(crossing))
    end
    #zeros(row_count, size(Z.G,2)+count(crossing))
    #Z.G .*= λ
    @timeit to "Set Matrix" begin
    Ĝ[:,1:size(Z.G,2)] .= Z.G
    Ĝ[crossing,size(Z.G,2)+1:end] .=  (@view I(row_count)[crossing, crossing])
    end
    @timeit to "Column Multiply" begin
    Ĝ[:,1:size(Z.G,2)] .*= λ
    Ĝ[:,size(Z.G,2)+1:end] .*= abs.(γ)
    end

    return Zonotope(Ĝ, ĉ, influence_new)
    end
end
