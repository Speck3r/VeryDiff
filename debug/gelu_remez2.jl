
using VeryDiff, Plots, SpecialFunctions

# CDF of N(0, 1)
# need to define it non-anonymously to differentiate
# between normal numbers and intervals
Φ(x::Number) = 0.5 * (1 + erf(x /sqrt(2)))

# gelu = x -> 0.5 * x * (1 + erf(x / sqrt(2)))
gelu = x -> x * Φ(x)
gelu_approx = x -> 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

dgelu = x -> Φ(x) + x * (1/sqrt(2π) * exp(-0.5*x^2))
d2gelu = x -> (1/sqrt(2π) * exp(-0.5*x^2)) * (2 - x^2)


function crit_overapprox_convex(f, df, d2f, l, u; n_iter=10, ∇tol=1e-10)
    x = 0.5 * (l + u)
    for i in 1:n_iter
        abs(df(x)) <= ∇tol && break
        x̂ = clamp(x - df(x)/d2f(x), l, u)
        if f(x̂) < f(x)
            x = x̂
        else
            # if Newton gets stuck then do gradient step
            # but make sure gradient delta is at most 1/10 of the width
            ∇ = clamp(df(x), -0.1*(u - l), 0.1*(u - l))
            x = clamp(x - ∇, l, u)
        end
    end

    # maximum must occur at the endpoints
    fᵤ = max(f(l), f(u))
    
    # every tangent is a valid lower bound
    # we choose to minimize the tangent at x
    # df(x) * (y - x) + f(x)
    # if df(x) >= 0, then min is at l, else at u
    x_star = ifelse(df(x) >= 0, l, u)
    y_star = df(x) * (x_star - x) + f(x)

    return y_star, fᵤ, x
end

function crit_overapprox_concave(f, df, d2f, l, u; n_iter=10, ∇tol=1e-10)
    gₗ, gᵤ, x_star = crit_overapprox_convex(x -> -f(x), x -> -df(x), x -> -d2f(x), l, u, n_iter=n_iter, ∇tol=∇tol)
    fₗ = -gᵤ
    fᵤ = -gₗ
    return fₗ, fᵤ, x_star
end

function crit_overapprox_convex(f, df, d2f, l, u, a; n_iter=10, ∇tol=1e-10)
    return crit_overapprox_convex(x -> f(x) - a*x, x -> df(x) - a, d2f, l, u, n_iter=n_iter, ∇tol=∇tol)
end

function crit_overapprox_concave(f, df, d2f, l, u, a; n_iter=10, ∇tol=1e-10)
    return crit_overapprox_concave(x -> f(x) - a*x, x -> df(x) - a, d2f, l, u, n_iter=n_iter, ∇tol=∇tol)
end

"""
Computes the critical points of GeLU(x) - a*x for x in [l, u] and an overapproximation of the function at those points.
"""
function gelu_crit_overapprox(l, u, a, b)
    xs_all = [l, u]
    ys_all = [gelu(l) - a*l, gelu(u) - a*u]
    
    if l <= -sqrt(2)
        l̂ = l
        û = min(u, -sqrt(2))
        fₗ, fᵤ, x_star = crit_overapprox_concave(gelu, dgelu, d2gelu, l̂, û, a)
        push!(xs_all, x_star)
        push!(ys_all, fᵤ)
    end
    if max(l, -sqrt(2)) <= min(u, sqrt(2))
        l̂ = max(l, -sqrt(2))
        û = min(u, sqrt(2))
        fₗ, fᵤ, x_star = crit_overapprox_convex(gelu, dgelu, d2gelu, l̂, û, a)
        push!(xs_all, x_star)
        push!(ys_all, fₗ)
    end
    if u >= sqrt(2)
        l̂ = max(l, sqrt(2))
        û = u
        fₗ, fᵤ, x_star = crit_overapprox_concave(gelu, dgelu, d2gelu, l̂, û, a)
        push!(xs_all, x_star)
        push!(ys_all, fᵤ)
    end

    return xs_all, ys_all .- b
end 


function convert_cheby_lin_to_monomial(p_cheby, l, u)
    # convert linear function in Chebyshev basis to monomial basis
    # just evaluate at 0 and 1 to get coeffs as
    # α = (y₁ - y₀)/(x₁ - x₀)
    # β = y₀  (since it was at 0)
    y₀ = VeryDiff.clenshaw_chebyshev(p_cheby, 0., l, u)
    y₁ = VeryDiff.clenshaw_chebyshev(p_cheby, 1., l, u)

    α = (y₁ - y₀)
    β = y₀  

    return [β, α]
end

function gelu_crit_overapprox_cheby(l, u, ps)
    a = convert_cheby_lin_to_monomial(ps, l, u)
    return gelu_crit_overapprox(l, u, a[2], a[1])
end



l, u = -5., 2.
degree = 1

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false
# VeryDiff.ROOTS_ALMOST_ZERO_TOL[] = 1e-15
p_lin, ϵ = VeryDiff.remez(gelu, (p, l, u) -> gelu_crit_overapprox_cheby(l, u, p), VeryDiff.poly_norm, l, u, degree, verbosity=1, max_iter=20)
f_lin = VeryDiff.make_eval_chebyshev(p_lin, l, u)

xs = range(l, u, 200)
plot(xs, gelu.(xs), label="gelu", framestyle=:origin)
plot!(xs, f_lin.(xs), label="lin approx", color=2)
plot!(xs, f_lin.(xs) .+ ϵ, label="upper", linestyle=:dash, color=2)
plot!(xs, f_lin.(xs) .- ϵ, label="lower", linestyle=:dash, color=2)

plot(xs, gelu.(xs) .- f_lin.(xs), label="error", framestyle=:origin)
