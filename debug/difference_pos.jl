
using VeryDiff, Plots


"""
Compute a linear relaxation of polynomials represented by a matrix of chebyshev coefficients.

The polynomial was initially fitted for the domain [l_fit, u_fit], the linear function approximates the polynomial in the input domain [l, u].
Approximation is computed using the Remez algorithm.

args:
    cs - chebyshev coefficients of the polynomials
    l_fit, u_fit - bounds for the original approximation domain of polynomial
    l, u - bounds for the input
"""
function get_linear_relaxation_cheby(cs::AbstractArray, l_fit::AbstractVector, u_fit::AbstractVector, l::AbstractVector, u::AbstractVector)
    row_count = length(l)
    nonlinmask = .~VeryDiff.islinear.(eachrow(cs))
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

    res = VeryDiff.approx_polynomial_lin.(eachrow(cs[nonlinmask, :]), l̂[nonlinmask], û[nonlinmask], l_fit[nonlinmask], u_fit[nonlinmask], max_iter=VeryDiff.REMEZ_ITERS[], cheby=true)
    λ[nonlinmask] .= getindex.(res, 1)  # slope of the input
    β[nonlinmask] .= getindex.(res, 2)  # bias
    γ[nonlinmask] .= getindex.(res, 3)  # new error

    # now need to transform from Chebyshev coefficients to α*x + β
    # so evaluate at x₀ = 0 and x₁ = 1 to get these coeffs as
    # α = (y₁ - y₀)/(x₁ - x₀)
    # β = y₀  (since it was at 0)
    y₀ = VeryDiff.clenshaw_chebyshev.(eachrow([β λ]), 0., l̂, û)
    y₁ = VeryDiff.clenshaw_chebyshev.(eachrow([β λ]), 1., l̂, û)

    α = (y₁ .- y₀)
    β = y₀

    return α, β, γ 
end


cs = VeryDiff.chebyshev_coefficients(x -> max(0, x), -5., 5., 4)
csx = VeryDiff.chebyshev_coefficients(x -> -x, -5., 5., 4)


function relax_pos_zonotope(cs, l_fit, u_fit, l, u)
    # compute chebyshev representation of -x for the same approximation domain as the given polynomials.
    # since -x is linear, we only need the first two coefficients and can thus always just use degree 1.
    id_neg = (l, u) -> VeryDiff.chebyshev_coefficients(x -> -x, l, u, 1)
    csx = hcat((id_neg.(l_fit, u_fit))...)'  # make it a matrix

    cs_diff = copy(cs)
    cs_diff[:, 1:2] .+= csx

    # compute the linear relaxation of the polynomial
    α, β, γ = get_linear_relaxation_cheby(cs_diff, l_fit, u_fit, l, u)

    return α, β, γ
end


cfun = (l, u) -> VeryDiff.chebyshev_coefficients(x -> max(0, x), l, u, 5)
l_fit = [-1, -3, 2, -3.]
u_fit = [1, -1, 5, 3.]
l = [-1, -2., 3, -1.5]
u = [1, -1.1, 4.5, 2.5]
cs = hcat((cfun.(l_fit, u_fit))...)'

α, β, γ = relax_pos_zonotope(cs, l_fit, u_fit, l, u)


idx = 1
xs = range(l_fit[idx], u_fit[idx], 200)
fc = VeryDiff.make_eval_chebyshev(cs[idx,:], l_fit[idx], u_fit[idx])
a = α[idx]
b = β[idx]
c = γ[idx]

csx = VeryDiff.chebyshev_coefficients(x -> -x, l_fit[idx], u_fit[idx], 5)
a1, b1, c1 = VeryDiff.approx_polynomial_lin(cs[idx,:] .+ csx, l[idx], u[idx], l_fit[idx], u_fit[idx], max_iter=VeryDiff.REMEZ_ITERS[], cheby=true)
flin = VeryDiff.make_eval_chebyshev([b1, a1], l[idx], u[idx])


plot(xs, fc.(xs) .- xs, label="f(x) - x", framestyle=:origin)
plot!(xs, a .* xs .+ b .- c, label="lb")
plot!(xs, a .* xs .+ b .+ c, label="ub")
vline!([l[idx], u[idx]])
plot!(xs, flin.(xs), label="lin")





