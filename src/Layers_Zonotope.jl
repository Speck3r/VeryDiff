
function (N::LayeredModel)(Z :: Zonotope, P :: PropState)
    return foldl((Z,L) -> L(Z,P),N.layers,init=Z)
end

function (N::LayeredModel)(Z::Zonotope, P::PropState, bounds::AbstractVector)
    foldl((Z,t) -> t[1](Z,P,bounds=t[2]), zip(N.layers, bounds), init=Z)
end

function (L::OXP.ONNXLinear)(Z :: Zonotope,P :: PropState; bounds=nothing)
    return @timeit to "Zonotope_DenseProp" begin
    G = L.dense.weight * Z.G
    c = L.dense.weight * Z.c .+ L.dense.bias
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

function (L::OXP.ONNXRelu)(Z :: Zonotope{N,GN,CN}, P :: PropState; bounds = nothing) where {N,GN,CN}
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
    
    @timeit to "Influence Matrix" begin
    if NEW_HEURISTIC
        # TODO(steuber): Can we avoid this reallocation?
        @timeit to "Allocation" begin
        #println(size(Z.influence,1), size(Z.influence,2)+count(crossing))
        influence_new = zeros(N, size(Z.influence,1), size(Z.influence,2)+count(crossing))
        end
        @timeit to "Set Matrix" begin
        influence_new[:,1:size(Z.influence,2)] .= Z.influence
        end
        # print("Hello")
        # print(size(influence_new))
        # print(size(Z.influence * Z.G[crossing,:]'))
        @timeit to "Multiply" begin
        influence_new[:,(size(Z.influence,2)+1):end] .=  abs.(Z.influence) * abs.(@view Z.G[crossing,:])'
        end
        # foreach(normalize!, eachcol(@view influence_new[:,(size(Z.influence,2)+1):end]))
    else
        influence_new = Z.influence
    end
    end

    @timeit to "Allocation" begin
    Ĝ = zeros(N,row_count, size(Z.G,2)+count(crossing))
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


"""
Compute a linear relaxation of polynomials represented by a matrix of chebyshev coefficients.

The polynomial was initially fitted for the domain [l_fit, u_fit], the linear function approximates the polynomial in the input domain [l, u].
Approximation is computed using the Remez algorithm.

args:
    cs - chebyshev coefficients of the polynomials
    l_fit, u_fit - bounds for the original approximation domain of polynomial
    l, u - bounds for the input

returns:
    α - slopes of the linear approximations in monomial basis (i.e. coefficient of x)
    β - biases of the linear approximations in monomial basis (i.e. coefficient of 1)
    γ - maximum errors of the linear approximations (s.t. α*x + β - ϵ ≤ p(x) ≤ α*x + β + ϵ)
"""
function get_linear_relaxation_cheby(cs::AbstractArray, l_fit::AbstractVector, u_fit::AbstractVector, l::AbstractVector, u::AbstractVector)
    row_count = length(l)
    nonlinmask = .~islinear.(eachrow(cs))
    l̂ = copy(l_fit)
    û = copy(u_fit)
    # only need to linearly approximate the nonlinear parts
    l̂[nonlinmask] .= l[nonlinmask]
    û[nonlinmask] .= u[nonlinmask]
    # for linear parts also the chebyshev coefficients are linear and thus only use the first two coefficients
    λ = copy(cs[:,2])
    β = copy(cs[:,1])
    # the error for linear functions is 0
    γ = zeros(row_count)

    res = approx_polynomial_lin.(eachrow(cs[nonlinmask, :]), l̂[nonlinmask], û[nonlinmask], l_fit[nonlinmask], u_fit[nonlinmask], max_iter=REMEZ_ITERS[], cheby=true)
    λ[nonlinmask] .= getindex.(res, 1)  # slope of the input
    β[nonlinmask] .= getindex.(res, 2)  # bias
    γ[nonlinmask] .= getindex.(res, 3)  # new error

    # now need to transform from Chebyshev coefficients to α*x + β
    # so evaluate at x₀ = 0 and x₁ = 1 to get these coeffs as
    # α = (y₁ - y₀)/(x₁ - x₀)
    # β = y₀  (since it was at 0)
    y₀ = clenshaw_chebyshev.(eachrow([β λ]), 0., l̂, û)
    y₁ = clenshaw_chebyshev.(eachrow([β λ]), 1., l̂, û)

    α = (y₁ .- y₀)
    β = y₀

    return α, β, γ 
end


function get_linear_relaxation(L::ONNXChebyshevPoly, lower, upper)
    get_linear_relaxation_cheby(L.coeffs, L.l, L.u, lower, upper)
end


function get_linear_relaxation(L::ONNXMonomialPoly, lower, upper)
    row_count = length(lower)
    nonlinmask = .~islinear.(eachrow(L.coeffs))
    λ = copy(L.coeffs[:,2])
    β = copy(L.coeffs[:,1])
    γ = zeros(row_count)

    # TODO is there a better way than eachrow()?
    res = approx_polynomial_lin.(eachrow(L.coeffs[nonlinmask, :]), lower[nonlinmask], upper[nonlinmask], max_iter=REMEZ_ITERS[], cheby=false)
    λ[nonlinmask] .= getindex.(res, 1)  # slope of the input
    β[nonlinmask] .= getindex.(res, 2)  # bias 
    γ[nonlinmask] .= getindex.(res, 3)  # new error
    
    return λ, β, γ
end


function (L::ONNXPoly)(Z::Zonotope{N,GN,CN}, P::PropState; bounds=nothing) where {N,GN,CN}
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
            λ, β, γ = get_linear_relaxation(L, lower, upper)
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
            Ĝ = zeros(N, row_count, size(Z.G, 2) + row_count)
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
