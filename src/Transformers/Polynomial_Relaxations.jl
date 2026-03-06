
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


function poly_initial_slope_guess(l₁, u₁, l₂, u₂, ∂l, ∂u)
    # currently only called, when l₂, u₂ is unstable
    # a is always 0
    # b has different cases 
    # l₁, u₁ negative: involves a₂
    # l₁, u₁ positive: involves λ₂ and coeff for Δ??? 
    # instable: 
    #   TODO: make it return a₁, a₂, aΔ          
    neg, pos, unstable = relu_stability_mask(l₁, u₁)
    a = zero(l₁)
    # TODO: what to do if ∂l == ∂u ?
    ∂λ = ifelse.((∂l .== 0) .& (∂u .== 0), 0., clamp.(∂u ./ (∂u .- ∂l), 0., 1.))
    b = ifelse.(unstable, ∂λ, 0.)
    #b = ifelse.(neg, .-u₂ ./ (u₂ .- l₂), ifelse.(pos, .-l₂ ./ (u₂ .- l₂), 0))
    return a, b
end


"""
Find linear relaxation for p(x) - ReLU(y), when y ≥ 0.

args:
    L - the current polynomial layer
    selector - indices or mask of the neurons where y ≥ 0 is true 
    lower₁ - concrete lower bounds on x
    upper₁ - concrete upper bounds on y 

returns:
    α, β, γ - s.t. α*x + β - γ ≤ p(x) - x ≤ α*x + β + γ
"""
function poly_pos_approx(L::ONNXChebyshevPoly, selector::AbstractVector, lower₁, upper₁)
    # compute chebyshev representation of -x for the same approximation domain as the given polynomials.
    # since -x is linear, we only need the first two coefficients and can thus always just use degree 1.
    id_neg = (l, u) -> VeryDiff.chebyshev_coefficients(x -> -x, l, u, 1)
    csx = hcat((id_neg.(L.l[selector], L.u[selector]))...)'  # make it a matrix

    cs_diff = copy(L.coeffs[selector,:])
    
    if count(selector) > 0
        # TODO: more elegant solution?
        # we could return earlier and don't hand it to get_linear_relaxation_cheby if there is no neuron to approximate.
        cs_diff[:, 1:2] .+= csx
    end

    get_linear_relaxation_cheby(cs_diff, L.l[selector], L.u[selector], lower₁[selector], upper₁[selector])
end