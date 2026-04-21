
# Reimplementation of Remez algorithm in Julia using Edoardo's Python code and the paper for Chebfun's implementation:
#   Pachon, Trefethen: Barycentric-Remez algorithms for best polynomial approximation in the chebfun system (2009)


"""
Chebyshev nodes in **closed** interval [l, u] of order n

Roots of chebyshev polynomial of the 2nd kind with degree n.
"""
function chebyshev_nodes(l, u, n)
    # why are they sorted the wrong way?
    sort([0.5*(l + u) + 0.5*(u - l)*cos(π*k / (n - 1)) for k in 0:n-1])
end


function chebyshev_approximation(f, l, u, n; kind=2)
    x = chebyshev_points(n, l, u, kind=kind)
    y = f.(x)
    
    B = x.^collect(0:n)'
    b = B \ y

    return b
end


"""
Chebyshev approximation for function that can handle vector input more efficiently than f.(x).
"""
function chebyshev_approximation_vecfun(f, l, u, n)
    @assert l != u "l == u!!! Can't solve singular system! l = $l, u = $u"
    x = chebyshev_nodes(l, u, n+2)
    y = f(x)
    
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
Returns a function evaluating the polynomial p(x) = p₀ + p₁x + p₂x² + ... in 
monomial basis.
"""
function make_eval_poly(ps::AbstractArray)
    return x -> sum(ps[k]*x^(k-1) for k in 1:length(ps))
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

    eval_poly = make_eval_poly(δpoly)

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
    eval_poly = make_eval_poly(ps)
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


function baryweights_chebfun(xs::AbstractVector{N}) where N<:Number
    # original matlab code: https://github.com/chebfun/chebfun/blob/master/baryWeights.m
    n = length(xs)
    C = 4/(maximum(xs) - minimum(xs))
    w = ones(N, n)
    for j = 1:n 
        v = C*(xs[j] .- xs)
        v[j] = 1.
        vv = exp(sum(log.(abs.(v))))
        w[j] = 1/(prod(sign.(v))*vv)
    end 

    return w ./ maximum(abs.(w))
end


function barycentric_weights(xs::AbstractVector{N}) where N<:Number
    # Code taken from https://tobydriscoll.net/fnc-julia/globalapprox/barycentric.html
    l = minimum(xs)
    u = maximum(xs)

    n = length(xs)

    #C = 4/(u - l)
    C = (u - l)/4

    xc = xs ./ C
    # Adding one node at a time, compute inverses of the weights.
    ω = ones(N, n)
    for m in 0:n-2
        d = xc[1:m+1] .- xc[m+2]    # vector of node differences
        @. ω[1:m+1] *= d            # update previous
        ω[m+2] = prod(-d)         # compute the new one
    end
    ws = 1 ./ ω 

    #ws = ones(n)
    #for j in 1:n 
    #    sign_prod = 1
    #    log_sum = 0.
    #    for v in 1:n 
    #        if v != j
    #            sign_prod *= sign(xs[j] - xs[v])
    #            log_sum += log(abs(xs[j] - xs[v]))
    #        end
    #    end
    #    
    #    ws[j] = sign_prod / exp(n * log(1/C) + log_sum)
    #end
    #
    #ws = ws / maximum(abs.(ws))
    return ws
end


function barycentric_interpolation(x::AbstractVector{N}, fval::AbstractVector{N}, xval::AbstractVector{N}, w::AbstractVector{N}) where N<:Number
    if length(fval) == 1
        # f is constant function
        return fill(fval, length(x))
    elseif any(isnan.(fval))
        return fill(N(NaN), length(x))
    end

    f = zeros(N, length(x))
    for j in 1:length(x)
        ŵ = w ./ (x[j] .- xval)
        f[j] = (ŵ' * fval) / sum(ŵ)
    end

    # fix NaN caused by division by zero
    idxs_x = 1:length(x)
    idxs_xval = 1:length(xval)
    for i in idxs_x[isnan.(f)]
        mask = xval .== x[i]
        if any(mask)
            f[i] = fval[idxs_xval[mask][1]]
        end
    end

    return f
end


"""
Computes next set of candidate points for the Remez algorithm.

args:
    xk - the current vector of candidate points 
    h  - the current levelled error 
    p  - coefficients of the current approximation polynomial in monomial form p₀ + p₁x + p₂x² + ...
    l  - concrete lower bound of approximation domain 
    u  - concrete upper bound of approximation domain 

returns:
    x_next - sorted vector of next candidate points 
    max_err - maximum absolute error between the true function and the candidate polynomial p
"""
function update_points(xk::AbstractVector{N}, h::N, p::AbstractVector{N}, f_error, l::N, u::N; method=:full_exchange) where N<:Number
    n = length(xk)
    x_error, y_error = f_error(p, l, u)
    abs_err = abs.(y_error)

    if method == :full_exchange
        xr = [x_error[abs_err .> h]; xk]
        y_error = y_error[abs_err .> h]
    elseif method == :one_point_exchange
        idx = minimum((1:length(x_error))[abs_err .>= maximum(abs_err)])
        xr = [x_error[idx]; xk]
        y_error = [y_error[idx]]
    else 
        @assert false "Update method $method not known, choose either :full_exchange or :one_point_exchange"
    end

    perm = sortperm(xr)
    xr = xr[perm]

    sigma = ones(N, length(xk))
    sigma[2:2:end] .= -one(N)

    err = [y_error; sigma .* h]
    err = err[perm]

    # delete repeated reference points and their error values
    # since they are sorted, same error points will be directly next to each other,
    # keep last point as points before will be thrown away if repeated.
    not_repeated = [xr[2:end] .- xr[1:end-1] .!= 0; true]
    xr = xr[not_repeated]
    err = err[not_repeated]

    x_next = [xr[1]]
    err_next = [err[1]]
    for i in 2:length(xr)
        # for adjacent values with same sign error, keep the one with larger absolute error
        if (sign(err[i]) == sign(err_next[end])) && (abs(err[i]) > abs(err_next[end]))
            x_next[end] = xr[i]
            err_next[end] = err[i]
        elseif sign(err[i]) != sign(err_next[end])
            # if the sign of the error changes, keep the point 
            push!(x_next, xr[i])
            push!(err_next, err[i])
        end
    end

    # TODO: this doesn't seem optimal, wouldn't we want to take the n largest errors of alternating sign?
    # choose degree+2 (==n) consecutive points that include the maximum error 
    idx = argmax(abs.(err_next))
    max_err = abs(err_next[idx])
    if n <= length(x_next)
        # if we can choose points (i.e. there are more than necessary)
        # take the points at most n to the left of the max
        d = max(idx - (n-1), 1)
        x_next = x_next[d:d+(n-1)]
    end 

    return x_next, max_err
end


"""
Function norm ||relu(x)|| = (∫ relu(x)ᵖ dx)^(1/p) over x in l..u 
"""
function relu_norm(l, u, p)
    return (1/(p+1) * u^(p+1))^(1/p)
end


function intpoly(ps)
    qs = zeros(length(ps)+1)
    qs[2:end] .= ps
    qs[2:end] .*= 1 ./ (1:length(ps))
    return qs
end


function poly_norm(l, u, ps)
    # TODO: this is not the function norm
    return norm(ps)
end


"""
Remez algorithm for finding the minimax polynomial approximation to a function f over the interval [l, u].

If max_iter==1, then the Chebyshev interpolation is computed.

Information:
    We start from the Chebyshev interpolation points.
    Then we compute the a function p(x) via barycentric interpolation (this function is guaranteed to be the 
    polynomial interpolating f at the given set of points xᵢ).
    Since p(x) is a polynomial, we can use Chebyshev interpolation (for the same degree) to get its Chebyshev coefficients.

args:
    f - function to approximate (continuous, Haar-condition)
    f_error - function f_error(ps, l, u) -> (x_error, y_error) returning extrema of f(x) - p(x), where 
              p is given as list of polynomial coefficients [p₀, p₁, ...]
    f_norm  - function f_norm(l, u, p) computing the p-norm of f (||∫ f(x)ᵖ dx||^(1/p) for x ∈ [l, u])
    l - concrete lower bound on approximation domain
    u - concrete upper bound on approximation domain
    degree - degree of approximation polynomial

kwargs:
    max_iter - maximum number of iterations
    verbosity
    tol - stop iterations, if (ϵ_max - h) / fnorm <= tol
    cheby - whether to use chebyshev or monomial form of polynomials (default: true)
"""
function remez(f, f_error, f_norm, l::N, u::N, degree::Integer; verbosity=0, max_iter=10, tol=N(1e-10), plotting=false, cheby=true) where N<:Number
    @assert l <= u "Approximation domain must be non-degenerate! Got [$l, $u]"
    @assert max_iter > 0 "max_iter > 0 required! Got $max_iter"

    # alternating signs
    sigma = ones(N, degree+2)
    sigma[2:2:end] .= -one(N)

    δ_best = N(Inf)
    p_best = zeros(N, degree+1)
    ϵ_best = N(Inf)
    x_best = zeros(N, degree+2)

    # get initial set of points
    #x_cur = VeryDiff.chebyshev_nodes(l, u, degree+2)
    x_cur = VeryDiff.chebyshev_points(degree+1, l, u)

    fnorm = f_norm(l, u, 2)

    verbosity > 0 && println("step, |error|,  |level|,  tol_diff, ref_diff")

    for i in 1:max_iter  
        f_cur = f.(x_cur)
        #w  = barycentric_weights(x_cur)
        w = baryweights_chebfun(x_cur)

        # levelled error 
        # i.e. error of equal magnitude and alternating sign at each x_cur
        h = (w' * f_cur) / (w' * sigma) 

        if h == 0
            h = N(1e-19)
        end

        # function values at x_cur for barycentric_interpolation
        p_cur = (f_cur .- h .* sigma)

        # cheby polynomial for barycentric_interpolation points
        if cheby
            p = chebyshev_coefficients_vec(x -> barycentric_interpolation(x, p_cur, x_cur, w), l, u, degree)
        else
            p = chebyshev_approximation_vecfun(x -> barycentric_interpolation(x, p_cur, x_cur, w), l, u, degree)
        end
        
        x_next, ϵ_max = update_points(x_cur, h, p, f_error, l, u)
        if ϵ_max / fnorm > 1e5
            x_next, ϵ_max = update_points(x_cur, h, p, f_error, l, u, method=:one_point_exchange)
            verbosity > 2 && println("\tONE_POINT_EXCHANGE")
        end

        if plotting
            xs = range(l, u, 100)
            plt = plot(xs, f.(xs), label="f(x)")
            #scatter!(x_cur, f_cur, label="x_$i")

            if cheby 
                plot!(xs, (x -> clenshaw_chebyshev(p, x, l, u)).(xs), label="p(x)")
            else
                plot!(xs, (x -> sum(p[k]*x^(k-1) for k in 1:length(p))).(xs), label="p(x)")
            end

            barys = barycentric_interpolation(xs, p_cur, x_cur, w)
            plot!(xs, barys, label="bary(x)")
            #scatter!(x_next, f.(x_next), label="x_$(i+1)")
            display(plt)
        end

        δ = ϵ_max - abs(h)
        Δx = maximum(abs.(x_next .- x_cur))
        if δ < δ_best
            p_best .= p 
            δ_best = δ
            ϵ_best = ϵ_max
            x_best .= x_next
        end

        verbosity > 0 && @printf("%-4d  %-5f  %-5f  %-8f  %-8f\n", i, ϵ_max, abs(h), δ/fnorm, Δx)
        verbosity > 1 && println("\tp = ", p)
        verbosity > 2 && println("\tx_cur = ", x_cur)

        if Δx <= 0
            verbosity > 0 && println("Reference points converged!")
            break 
        elseif δ/fnorm <= tol
            verbosity > 0 && println("Tolerance of $tol reached (", δ/fnorm, ")!")
            break
        end

        x_cur = x_next
    end

    return p_best, ϵ_best
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
#function remez(f, f_error, l, u, degree; max_iter=10, verbosity=0, opt_tol=1.05)
#    x = chebyshev_nodes(l, u, degree+2)
#    y = f.(x)
#
#    p̂ = zeros(degree + 1)
#    ϵ = Inf
#    for i in 1:max_iter
#        B = x.^collect(0:degree)'
#        B̂ = [B (-1).^collect(1:degree+2)]    
#        b = B̂ \ y
#        
#        p̂ = b[1:end-1]
#        ϵ = b[end]
#
#        x_error, y_error = f_error(p̂, l, u)
#        ϵ = maximum(abs.(y_error))
#
#        if max_iter <= 1
#            verbosity > 0 && println("Chebyshev approximation! |error| = ", ϵ)
#            # only chebyshev approximation 
#            # processing for further iterations is not needed
#            return p̂, ϵ
#        end
#
#
#       # TODO: what if we have less than degree+2 extrema? (can happen if we try to approx a linear function by a higher order poly)
#        # we want the degree+2 extrema of alternating sign with largest possible absolute value 
#        # TODO: this may not find the best alternating sequence as we only look at consecutive alternating elements
#        windows = [minimum(abs.(y_error[j:j+degree+1])) for j in 1:length(y_error)-(degree+1)]
#
#        @assert length(windows) > 0 "Did not find degree+2 extrema. A common cause of failure is that IMAG_TOL[] is too low."
#
#        window_idx = argmax(windows)
#        #window_idx = argmax([minimum(abs.(y_error[j:j+degree+1])) for j in 1:length(y_error)-(degree+1)])
#        x_error = x_error[window_idx:window_idx+degree+1]
#        y_error = y_error[window_idx:window_idx+degree+1]
#
#        x = x_error
#        y = f.(x)
#        #ϵ = maximum(abs.(y_error))
#
#        verbosity > 0 && println(i, ": |error| = ", ϵ)
#        verbosity > 1 && println("\terrors = ", y_error)
#
#        if ϵ <= opt_tol * minimum(abs.(y_error))
#            verbosity > 0 && println("\toptimality tol (", opt_tol, ") reached!")
#            break
#        end
#    end    
#
#    return p̂, ϵ
#end     


"""
Finds best linear approximation to polynomial given by coefficients ps via the Remez algorithm.

The original polynomial might also approximate some function over the interval [l̂, û].
And (in case of the Chebyshev approximation) the coefficients might depend on that range. 
So these can also be specified. 

args:
    ps - coefficients of the polynomial to approximate (either monomial or Chebyshev coefficients for increasing degree)
    l  - lower bound for domain of linear approximation 
    u  - upper bound for domain of linear approximation 

default args: (they can be ignored if dealing with polynomials in monomial basis)
    l̂ - lower bound for approximation domain of the original polynomial p(x) (default -1)
    û - upper bound for approximation domain of the original polynomial p(x) (default  1)

kwargs:
    verbosity - verbosity argument for Remez algorithm
    tol - stopping tolerance for Remez algorithm
    max_iter - maximum number of Remez iterations to find the linear approximation
    cheby - whether polynomial is given in Chebyshev basis (in this case also the linear approximation will be returned in Chebyshev basis)

returns:
    α - coefficient of x in monomial form or coefficient of T₁(x) for scaling to x ∈ [l, u] for chebyshev form 
    β - coefficient of 1 in monomial form or coefficient of T₀(x) for scaling to x ∈ [l, u] for chebyshev form
    ϵ - maximum approximation error of the linear function given by α and β
"""
function approx_polynomial_lin(ps::AbstractVector{N}, l::N, u::N, l̂=-one(N), û=one(N); verbosity=0, tol=N(1e-10), max_iter=10, cheby=true) where N<:Number
    # println("l = $l, u = $u, l̂ = $l̂, û = $û, ps = $ps")
    if cheby 
        # need to get polynomials to common domain, s.t. we can just add and subtract the coefficient vectors.
        # anonymous function value changes, when we change ps later on !!! (see https://discourse.julialang.org/t/anonymous-functions-and-overwriting-arguments/127586)
        # poly = x -> clenshaw_chebyshev(ps, x, l̂, û)
        poly = make_eval_chebyshev(ps, l̂, û)

        if (l̂ == û) && all(ps .== 0)
            # how can that happen?
            # TODO: better solution than just an if?
            # poly(l) == 0 in this case, that's why it works!
            return zero(N), poly(l), zero(N)
        elseif l == u 
            return zero(N), poly(l), zero(N)
        end

        degree = length(ps) - 1
        # need to normalize polynomial to x ∈ [l, u]
        ps = chebyshev_coefficients(poly, l, u, degree)

        errfun = (p, l, u) -> poly_error_cheby(ps, p, l, u)
    else  
        poly = make_eval_poly(ps)

        if l == u 
            # how can that happen?
            # TODO: better solution than just an if?
            return zero(N), poly(y), zero(N)
        end

        errfun = (p, l, u) -> poly_error(ps, p, l, u)
    end 

    p_lin, ϵ = remez(poly, errfun, poly_norm, l, u, 1, verbosity=verbosity, tol=tol, max_iter=max_iter, cheby=cheby)
    β, α = p_lin

    return α, β, ϵ
end


function approx_relu_poly(l::N, u::N, degree::Integer; verbosity=0, tol=N(1e-10), max_iter=10, plotting=false, cheby=true) where N<:Number
    f = x -> max.(0, x)

    if u <= 0
        # we can just set everything to zero in both the chebyshev and the monomial case
        p = zeros(N, degree+1)
        ϵ = zero(N)
    elseif l >= 0
        @assert degree > 0 "ReLU approximation currently not implemented for degree = 0 for fixed active case!"

        if cheby 
            # although T₁(x) = x, we cannot just set p[2] = 1 in the Chebyshev case because it is evaluated 
            # w.r.t x ∈ [l, u], i.e. that would be T₁((x - 0.5(l + u))/(0.5 * (u - l))) ≠ x
            # only need degree 1 for linear functions
            cs = chebyshev_coefficients(f, l, u, 1)
            p = zeros(N, degree+1)
            p[1:2] .= cs
        else        
            p = zeros(N, degree+1)
            p[2] = one(N)
        end
        ϵ = zero(N)
    else
        if cheby
            p, ϵ = remez(f, relu_error_cheby, relu_norm, l, u, degree, verbosity=verbosity, tol=tol, max_iter=max_iter, plotting=plotting, cheby=true)
        else
            p, ϵ = remez(f, relu_error, relu_norm, l, u, degree, verbosity=verbosity, tol=tol, max_iter=max_iter, plotting=plotting, cheby=false)
        end
    end

    return p, ϵ
end


function approx_gelu_poly(l::N, u::N, degree::Integer; verbosity=0, tol=N(1e-10), max_iter=10, plotting=false, cheby=true) where N<:Number
    @assert cheby "GeLU Remez approximation is only defined for Chebyshev basis."
    if u <= GELU_PP.a
        # just use the linear approximation gelu(x) ≈ 0 for x <= a
        p = zeros(N, degree+1)
        ϵ = zero(N)
    elseif l >= GELU_PP.b
        # just use the linear approximation gelu(x) ≈ x for x >= b
        # TODO: is there a better way to get the Chebyshev coeffs for f(x) = x?
        f_lin = x -> x
        cs = VeryDiff.chebyshev_coefficients(f_lin, l, u, 1)
        p = zeros(N, degree + 1)
        p[1:2] .= cs 
        ϵ = zeros(N)
    else
        f_gpp = make_eval_gelu_piecewise_poly(GELU_PP)
        errfun = (p, l, u) -> piecewise_poly_error(GELU_PP, p, l, u)
        p, ϵ = remez(f_gpp, errfun, poly_norm, l, u, degree, verbosity=verbosity, tol=tol, max_iter=max_iter, plotting=plotting, cheby=cheby)
    end

    return p, ϵ
end