


function chebyshev_points(degree::Integer, l=-1, u=1; kind=2)
    if kind == 1
        xs = [cos(π*(k + 0.5)/degree) for k = 0:degree]
    elseif kind == 2
        xs = [cos(k*π / degree) for k = 0:degree]
    else 
        throw(ArgumentError("There are no Chebyshev points of the $kind kind! Only 1 or 2."))
    end

    return 0.5*(u-l) .* xs .+ 0.5*(u+l)
end

"""
Computes chebyshev coefficients for the polynomial approximation of function f.

The Chebyshev coefficients are **always** for f(x) ≈ c₀T₀(x) + c₁T₁(x) + ... i.e. for Chebyshev polynomials of the 1st kind,
regardless of interpolation was w.r.t. Chebyshev points of the 1st or 2nd kind.

Interpolation w.r.t. Chebyshev points of the 1st kind leads to the error being spread out more evenly over all of [-1, 1],
whereas interpolation w.r.t. Chebyshev points of the 2nd kind ensures that f(-1) and f(1) are preserved exactly.

args:
    f - function to approximate
    degree - degree of the polynomial approximation 

kwargs:
    kind - whether to use Chebyshev points of the 1st or 2nd kind for interpolation (defaults to 2)

returns:
    cs - Chebyshev coefficients for f(x) ≈ p(x) = c₀T₀(x) + c₁T₁(x) + ...
"""
function chebyshev_coefficients(f, degree::Integer; kind=2)
    N = degree + 1
    if kind == 1
        # xk = [cos(π*(k + 0.5)/N) for k = 0:N-1]
        cs = [2/N * sum([f(cos(π*(k + 0.5) / N)) * cos(π*j*(k + 0.5) / N) for k =0:N-1]) for j = 0:N-1]

        # f(x) ≈ (2/n * ∑ₙ cₙ⋅Tₙ(x)) - 0.5*c₀
        # since T₀(x) = 1, we can just adjust that coefficient
        cs[1] -= 0.5*cs[1] 
    else
        # xk = [cos(k*π / degree) for k = 0:degree]
        # coded based on https://andrea-combette.com/post/spectral-chebyshex/
        # why are there so few resources on 2nd kind Chebyshev interpolation?
        c̄ = ones(degree + 1)
        c̄[1]   = 2
        c̄[end] = 2
        cs = [2/(degree*c̄[j+1]) * sum(f(cos(k*π/degree)) * cos(k*j*π/degree)/c̄[k+1] for k = 0:degree) for j = 0:degree]
    end
    cs
end


"""
Computes Chebyshev coefficients for the polynomial approximation of function f.
If you use this function, f should be able to handle vector input more efficiently than f.(x) !

See also documentation of `chebyshev_coefficients()`

args:
    f - function to approximate (should be able to handle vector input more efficiently than f.(x))
    degree - degree of the polynomial approximation

kwargs:
    kind - whether to use Chebyshev points of the 1st or 2nd kind for interpolation (defaults to 2)

returns: 
    cs - Chebyshev coefficients form of f(x) ≈ p(x) = c₀T₀(x) + c₁T₁(x) + ...
"""
function chebyshev_coefficients_vec(f, degree::Integer; kind=2)
    xk = chebyshev_points(degree, kind=kind)
    fk = f(xk)
    if kind == 1
        N = degree+1
        cs = [2/N * sum([fk[k+1] * cos(π*j*(k + 0.5) / N) for k = 0:degree]) for j = 0:degree]
    else
        c̄ = ones(degree + 1)
        c̄[1]   = 2
        c̄[end] = 2
        cs = [2/(degree*c̄[j+1]) * sum(fk[k+1] * cos(k*j*π/degree)/c̄[k+1] for k = 0:degree) for j = 0:degree]
    end

    return cs    
end


function chebyshev_coefficients(f, l::N, u::N, degree::Integer; kind=2) where N<:Number
    f̂ = x -> f(0.5 * (u - l)*x + 0.5*(u + l))
    cs = chebyshev_coefficients(f̂, degree, kind=kind)
end


function chebyshev_coefficients_vec(f, l::N, u::N, degree::Integer; kind=2) where N<:Number 
    f̂ = x -> f(0.5 .* (u .- l) .* x .+ 0.5 .* (u .+ l))
    cs = chebyshev_coefficients_vec(f̂, degree, kind=kind)   
end


"""
Evaluates polynomial with Chebyshev coefficients cs and approximation domain [l, u] at point x 
using Clenshaw recurrence.

args:
    cs - Chebyshev coefficients for p(x) = c₀T₀(x) + c₁T₁(x) + ...
    x - point at which to evaluate p 
    l - lower bound of the approximation domain 
    u - upper bound of the approximation domain 

kwargs:
    printing - print values of u_k, u_{k+1} and u_{k+2} at each iteration.

returns:
    p(x) - value of p at input x
"""
function clenshaw_chebyshev(cs, x, l=-1, u=1; printing=false)
    # normalize to [-1, 1]
    x = (x - 0.5*(u+l)) / (0.5*(u-l))

    n = length(cs) - 1 # because we have length(cs) = degree + 1

    u  = 0.
    u1 = 0.  # u_{k+1}
    u2 = 0.  # u_{k+2}
    for i in 0:(n-1) 
        k = n - i
        # need cs[k+1] because of 1-based indexing
        u = 2*x*u1 - u2 + cs[k+1]

        printing && println(k, ": u = ", u, ", u1 = ", u1, ", u2 = ", u2)

        u2 = u1
        u1 = u 
    end

    u = x*u1 - u2 + cs[1]
end


function make_eval_chebyshev(cs::AbstractVector, l=-1, u=1)
    p = x -> clenshaw_chebyshev(cs, x, l, u)
end


"""
Constructs function p(x) that computes the values of the chebyshev approximation of f(x) over x ∈ [l, u]
"""
function make_eval_chebyshev(f, l::N, u::N, degree::Integer; kind=2) where N<:Number 
    cs = chebyshev_coefficients(f, l, u, degree, kind=kind)
    polyfun = make_eval_chebyshev(cs, l, u)
end


function normalize_chebyshev(cs::AbstractVector, l::N, u::N) where N<:Number
    degree = length(cs) - 1
    fc = x -> clenshaw_chebyshev(cs, x, l, u)
    coeffs_normalized = chebyshev_coefficients.(fc, -1, 1, degree)  
    return coeffs_normalized
end


function colleague_matrix(cs::AbstractVector)
    n = length(cs)
    @assert abs(cs[n]) >= 1e-12 "Colleague matrix is only possible for full degree Chebyshev polynomials, but last coeff is almost zero: $(cs[end])"
    # need case distinction because construct off-diagonals with ones(n-2) and ones(n-3) which would be negative otherwise.
    if n == 2
        T̂ = Matrix([0.;;])
    elseif n == 3
        T̂ = Matrix([0. 1; 0.5 0])
    else
        dl = 0.5 .* ones(n-2)
        d  = zeros(n-1)
        du = [1.; 0.5 .* ones(n-3)]
        T = Tridiagonal(dl, d, du)    

        # can we avoid that?
        T̂ = Matrix(T)
    end 

    T̂[end,:] .-= 1/(2*cs[end]) .* cs[1:end-1]
    T̂
end


"""
Computes the roots of a polynomial p(x) in Chebyshev basis.

Root finding is done via reduction to eigenvalues of Colleague matrices 
(cf. chapter 18 in Trefethen: Approximation Theory and Approximation Practice, 2018)

The interval [l, u] describes the **approximation domain** over which p(x) approximates some function.
We do NOT restrict the root finding to this interval!

args:
    cs - vector of Chebyshev coefficients 
    l - lower bound on approximation domain of p(x) (default -1)
    u - upper bound on approximation domain of p(x) (default 1)
"""
function chebyshev_roots(cs::AbstractVector{N}, l=-1, u=1) where N<:Number
    if length(cs) <= 1
        # a constant polynomial has either zero or infinitely many roots.
        return Vector{N}()
    else
        Cm = colleague_matrix(cs)
        vals = eigvals(Cm)

        rs = real.(vals[abs.(imag.(vals)) .< IMAG_TOL[]])
        return rs * 0.5*(u - l) .+ 0.5*(u + l)
    end
end



"""
Computes the chebyshev coefficients of d/dx p(x), where p has Chebyshev coefficients cs.

If p(x) has Chebyshev coefficients c₀T₀(x) + ... + cₙTₙ(x), the coefficients cᵢ' of the derivative are computed using the formula
    cₙ₊₁' = 0
    cₙ'   = 0
    cₖ'   = 2(k+1)cₖ₊₁ + cₖ₊₂' (for 1 < k < n)
    c₀'   = c₁ + (1/2) c₂'

args:
    cs - Chebyshev coefficients s.t. p(x) = c₀T₀(x) + c₁T₁(x) + ...

returns:
    cp - Chebyshev coefficients s.t. d/dx p(x) = cp₀T₀(x) + cp₁T₁(x) + ...
"""
function chebyshev_derivative(cs::AbstractVector)
    # c_0' = c_1 + (1/2) c_2'
    # c_n' = 2(n+1)c_{n+1} + c_{n+2}'  (if n > 0)
    # c_n' = 0 if original function only had degree n
    cp = zeros(length(cs)+1)
    n = length(cs)
    for i in 1:(n-2)
        k = n - i 
        # keep in mind that cp[k] contains c_{k-1}' and not c_k' (1-indexing in Julia)
        cp[k] = cp[k+2] + 2*k*cs[k+1]
    end
    cp[1] = cs[2] + 0.5 * cp[3]

    cp[1:end-2]
end


"""
Computes extrema of ReLU(x) - p(x) for a polynomial in **Chebyshev representation** 
over the interval [l, u].

args:
    cs - Chebyshev coefficients c₀T₀(x) + c₁T₁(x) + c₂T₂(x) + ... of the polynomial p 
    l - concrete lower bound of the interval 
    u - concrete upper bound of the interval

returns:
    xs - sorted locations of the extrema 
    ys - values of ReLU(x) - p(x) at the extrema
"""
function relu_error_cheby(cs::AbstractVector, l=-1, u=1)
    # TODO: do we really want to evaluate here? Or rather in Remez algorithm?
    eval_poly = make_eval_chebyshev(cs, l, u)
    errfun = x -> max.(0, x) - eval_poly(x)

    # need Chebyshev coefficients of f(x) = x over [l,u]
    cx = chebyshev_coefficients(x -> x, l, u, 1)
    dx = chebyshev_derivative(cx)

    # errfun(x) = ReLU(x) - p(x), so want derivative of negative of poly
    cp = chebyshev_derivative(.-cs)

    # case 1: ReLU(x) = 0
    # -> errfun(x) = -p(x)
    #    only need extrema of -p(x) in [l, 0]
    xs_zero = chebyshev_roots(cp, l, u)
    xs_zero = [x for x in xs_zero if (l <= x) && (x <= 0)]

    # case 2: ReLU(x) = x
    # -> errfun(x) = x - p(x)
    cp[1] += dx[1]
    xs_one = chebyshev_roots(cp, l, u)
    xs_one = [x for x in xs_one if (0 <= x) && (x <= u)]

    boundary = (l < 0) & (u > 0) ? [l, 0, u] : [l, u]
    xs = [xs_zero; xs_one; boundary]
    perm = sortperm(xs)
    xs = xs[perm]
    ys = errfun.(xs)

    return xs, ys
end


"""
Computes extrema of polynomials p(x) - q(x) in Chebyshev representation over the interval [l, u].

The polynomials p(x) and q(x) must be in Chebyshev represenation with the **SAME APPROXIMATION DOMAIN** [l, u].
This is necessary, s.t. we can just subtract coefficients of both polynomials in Chebyshev basis.

args:
    ps - Chebyshev coefficients of the polynomial to be approximated in order [p₀, p₁, ...]
    qs - Chebyshev coefficients of the approximating polynomial in order [q₀, q₁, ...]
    l - concrete lower bound of the approximation interval
    u - concrete upper bound of the approximation interval

returns:
    xs - locations of the extrema
    ys - values of p(x) - q(x) at the extrema
"""
function poly_error_cheby(ps, qs, l, u)
    # polynomial representing p(x) - q(x)
    δpoly = zeros(max(length(ps), length(qs)))
    δpoly[1:length(ps)] .+= ps
    δpoly[1:length(qs)] .-= qs

    eval_poly = make_eval_chebyshev(δpoly, l, u)

    dδpoly = chebyshev_derivative(δpoly)
    xs = chebyshev_roots(dδpoly, l, u)
    xs = [x for x in xs if (l <= x) && (x <= u)]
    xs = [xs; [l, u]]
    perm = sortperm(xs)
    xs = xs[perm]
    ys = eval_poly.(xs)

    return xs, ys
end