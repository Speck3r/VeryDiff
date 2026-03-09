
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
    constant_mask = l .== u 

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

    a = zeros(row_count)
    b = zeros(row_count)
    # now need to transform from Chebyshev coefficients to α*x + β
    # so evaluate at x₀ = 0 and x₁ = 1 to get these coeffs as
    # α = (y₁ - y₀)/(x₁ - x₀)
    # β = y₀  (since it was at 0)
    y₀ = clenshaw_chebyshev.(eachrow([β λ])[.~constant_mask], 0., l̂[.~constant_mask], û[.~constant_mask])
    y₁ = clenshaw_chebyshev.(eachrow([β λ])[.~constant_mask], 1., l̂[.~constant_mask], û[.~constant_mask])

    a[.~constant_mask] .= (y₁ .- y₀)
    b[.~constant_mask] .= y₀

    # fitting a linear function for degenerate interval [l,u] with l == u fails, so we just evaluate at the single point
    # this incurs no error
    a[constant_mask] .= zero(eltype(cs))
    b[constant_mask] .= clenshaw_chebyshev.(eachrow(cs)[constant_mask], l[constant_mask], l_fit[constant_mask], u_fit[constant_mask])

    return a, b, γ 
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