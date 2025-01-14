
"""
Chebyshev nodes in **closed** interval [l, u] of order n

Roots of chebyshev polynomial of the 2nd kind with degree n.
"""
function chebyshev_nodes(l, u, n)
    # why are they sorted the wrong way?
    sort([0.5*(l + u) + 0.5*(u - l)*cos(π*k / (n - 1)) for k in 0:n-1])
end


function chebyshev_approximation(f, l, u, n)
    x = chebyshev_nodes(l, u, n+2)
    y = f.(x)
    
    B = x.^collect(0:n)'
    B̂ = [B (-1).^collect(1:n+2)]    
    b = B̂ \ y
    b = b[1:end-1]
    ϵ = b[end]

    return b
end


"""
Coefficients of derivative of univariate polynomial

args:
    ps - coefficients of the polynomial in order [p₀, p₁, ...]
"""
function dpoly(ps)
    dps = [i*ps[i+1] for i in 1:length(ps)-1]
    return dps
end


"""
Computes all real-valued roots of a polynomial p(x).

args: 
    ps - coefficients of the polynomial in order [p₀, p₁, ...]
"""
function real_roots(ps)
    @assert any(ps .!= 0) "Zero polynomial has roots everywhere! PolynomialRoots is not able to handle that!"
    rs = roots(ps)
    return real.(rs[abs.(imag.(rs)) .< IMAG_TOL[]])
end


function ensure_alternating_signs(ys)
    n_sing_changes = count(ys[1:end-1] .* ys[2:end] .< 0)
    inds = zeros(Int, n_sing_changes + 1)
    cnt = 1
    y_prev = ys[1]
    prev_ind = 1
    for (i, y) in enumerate(ys[2:end])
        if sign(y_prev) == sign(y)
            if abs(y) > abs(y_prev)
                prev_ind = i+1  # because we started at 2
                y_prev = y 
            end
        else
            inds[cnt] = prev_ind
            cnt += 1
            y_prev = y
            prev_ind = i+1
        end
    end

    inds[end] = prev_ind

    return inds
end


"""
Computes extrema of polynomials p(x) - q(x) over the interval [l, u].

If two consecutive extrema have the same sign, keep the one with larger absolute value.

args:
    ps - coefficients of the polynomial to be approximated in order [p₀, p₁, ...]
    qs - coefficients of the approximating polynomial in order [q₀, q₁, ...]
    l - concrete lower bound of the approximation interval
    u - concrete upper bound of the approximation interval

returns:
    xs - locations of the extrema
    ys - values of p(x) - q(x) at the extrema
"""
function poly_error(ps, qs, l, u)
    # polynomial representing p(x) - q(x)
    δpoly = zeros(max(length(ps), length(qs)))
    δpoly[1:length(ps)] .+= ps
    δpoly[1:length(qs)] .-= qs

    eval_poly = x -> sum(δpoly[k]*x^(k-1) for k in 1:length(δpoly))

    dδpoly = dpoly(δpoly)
    xs = real_roots(dδpoly)
    xs = [x for x in xs if (l <= x) && (x <= u)]
    xs = [xs; [l, u]]
    perm = sortperm(xs)
    xs = xs[perm]
    ys = eval_poly.(xs)

    inds = ensure_alternating_signs(ys)

    return xs[inds], ys[inds]
end


"""
Computes extrema of ReLU(x) - p(x) for a polynomial p over the interval [l, u].

args:
    ps - coefficients [p₀, p₁, ...] of the polynomial p
    l - concrete lower bound of the interval
    u - concrete upper bound of the interval

returns:
    xs - sorted locations of the extrema
    ys - values of ReLU(x) - p(x) at the extrema
"""
function relu_error(ps, l::N, u::N) where N<:Number
    dps = dpoly(.-ps)
    eval_poly = x -> sum(ps[k]*x^(k-1) for k in 1:length(ps))
    errfun = x -> max.(0, x) - eval_poly(x)

    # case 1: ReLU(x) = 0
    # -> errfun(x) = -p(x)
    #    only need extrema of -p(x) in [l, 0]
    xs_zero = real_roots(dps)
    xs_zero = [x for x in xs_zero if (l <= x) && (x <= 0)]

    # case 2: ReLU(x) = x
    # -> errfun(x) = x - p(x)
    #    need extrema of x - p(x) in [0, u]
    dps[1] += 1
    xs_one = real_roots(dps)
    xs_one = [x for x in xs_one if (0 <= x) && (x <= u)]

    boundary = (l < 0) & (u > 0) ? [l, 0, u] : [l, u]
    xs = [xs_zero; xs_one; boundary]
    perm = sortperm(xs)
    xs = xs[perm]
    ys = errfun.(xs)

    return xs, ys
end 


"""
Remez algorithm for finding the minimax polynomial approximation to a function f over the interval [l, u].

If max_iter==1, then the Chebyshev interpolation is computed.

args:
    f - function to approximate (continuous, Haar-condition)
    f_error - function f_error(ps, l, u) -> (x_error, y_error) returning extrema of p(x) - f(x), where 
              p is given as list of polynomial coefficients [p₀, p₁, ...]
    l - concrete lower bound on approximation domain
    u - concrete upper bound on approximation domain
    degree - degree of approximation polynomial

kwargs:
    max_iter - maximum number of iterations
    verbosity
    opt_tol - stop iterations, if ϵ_max <= opt_tol * ϵ_min
"""
function remez(f, f_error, l, u, degree; max_iter=10, verbosity=0, opt_tol=1.05)
    x = chebyshev_nodes(l, u, degree+2)
    y = f.(x)

    p̂ = zeros(degree + 1)
    ϵ = Inf
    for i in 1:max_iter
        B = x.^collect(0:degree)'
        B̂ = [B (-1).^collect(1:degree+2)]    
        b = B̂ \ y
        
        p̂ = b[1:end-1]
        ϵ = b[end]

        x_error, y_error = f_error(p̂, l, u)
        ϵ = maximum(abs.(y_error))
        # TODO: what if we have less than degree+2 extrema? (can happen if we try to approx a linear function by a higher order poly)
        # we want the degree+2 extrema of alternating sign with largest possible absolute value 
        # TODO: this may not find the best alternating sequence as we only look at consecutive alternating elements
        window_idx = argmax([minimum(abs.(y_error[j:j+degree+1])) for j in 1:length(y_error)-(degree+1)])
        x_error = x_error[window_idx:window_idx+degree+1]
        y_error = y_error[window_idx:window_idx+degree+1]

        x = x_error
        y = f.(x)
        #ϵ = maximum(abs.(y_error))

        verbosity > 0 && println(i, ": |error| = ", ϵ)
        verbosity > 1 && println("\terrors = ", y_error)

        if ϵ <= opt_tol * minimum(abs.(y_error))
            verbosity > 0 && println("\toptimality tol (", opt_tol, ") reached!")
            break
        end
    end    

    return p̂, ϵ
end     


function approx_polynomial_lin(ps, l, u; verbosity=0, opt_tol=1.01, max_iter=10)
    poly = x -> sum(ps[k]*x^(k-1) for k in 1:length(ps))
    p_lin, ϵ = remez(poly, (p, l, u) -> poly_error(ps, p, l, u), l, u, 1, verbosity=verbosity, opt_tol=opt_tol, max_iter=max_iter)
    β, α = p_lin
    return α, β, ϵ
end