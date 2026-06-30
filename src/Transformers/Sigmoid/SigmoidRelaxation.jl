
"""
    `crit_overapprox_convex(f, df, d2f, l, u; n_iter=10, ∇tol=1e-10)`

    Computes critical points of a convex function `f` on the interval `[l, u]` and an overapproximation of the function at those points. 
    
    The overapproximation is computed using the tangent at the candidate critical point found via Newton's method. 
    Since any tangent is a lower bound, we can minimize the tangent over [l, u] to get a valid overapproximation. 

    args:
    -  f - the function for which to compute the critical points and overapproximation
    -  df - the derivative of f
    -  d2f - the second derivative of f
    -  l - the lower bound of the interval
    -  u - the upper bound of the interval

    kwargs:
    -  n_iter - the maximum number of iterations for Newton's method
    -  ∇tol - the tolerance for the gradient to consider a critical point found

    returns:
    -  fₗ - the overapproximation of f at the critical point (lower bound)
    -  fᵤ - the maximum of f over the interval (upper bound)
    -  x_star - the critical point candidate for fₗ
"""
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

"""
    `gelu_crit_overapprox_cheby(l, u, ps)`

    Computes the critical points and an overapproximate function values of of GeLU(x) - p(x) for x in [l, u] where p is a *linear* function in Chebyshev basis.
        
    args:
    -  l - the lower bound of the interval
    -  u - the upper bound of the interval
    -  ps - Chebyshev coefficients of the linear function in order [p₀, p₁]
    
    returns:
    -  xs - the critical points found
    -  ys - overapproximate values of GeLU(x) - p(x) at the critical points
"""
function gelu_crit_overapprox_cheby(l, u, ps)
    a = convert_cheby_lin_to_monomial(ps, l, u)
    return gelu_crit_overapprox(l, u, a[2], a[1])
end


function approx_gelu_lin(l::N, u::N; verbosity=0, tol=N(1e-10), max_iter=10, cheby=true) where N<:Number
    if l == u 
        # if l == u, we cannot use Chebyshev approximation as we usually normalize to [-1, 1] which requires division by u-l
        return zero(N), gelu(l), zero(N)
    end

    degree = 1
    p_lin, ϵ = remez(gelu, (p, l, u) -> gelu_crit_overapprox_cheby(l, u, p), VeryDiff.poly_norm, l, u, degree, 
                              verbosity=verbosity, max_iter=max_iter, tol=tol, cheby=cheby)
                                
    a = convert_cheby_lin_to_monomial(p_lin, l, u)
    return a[2], a[1], ϵ
end

function get_linear_relaxation(L::ONNXGelu, lower, upper)
    res = approx_gelu_lin.(lower, upper, max_iter=REMEZ_ITERS[], cheby=true)
    λ = getindex.(res, 1)
    β = getindex.(res, 2)
    γ = getindex.(res, 3)
    return λ, β, γ
end