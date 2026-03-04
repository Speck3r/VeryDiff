
"""
Represents a univariate polynomial p(x) = c₀T₀(x) + c₁T₁(x) + ... + cₙTₙ(x) over the interval x ∈ [l, u]
in Chebyshev basis.
"""
struct ChebyshevPolynomial{N<:Number,CN<:AbstractVector{N}} 
    coeffs::CN
    l::N 
    u::N
end


Base.size(c::ChebyshevPolynomial) = size(c.coeffs)
Base.length(c::ChebyshevPolynomial) = length(c.coeffs)


function Base.copy(c::ChebyshevPolynomial)
    return ChebyshevPolynomial(copy(c.coeffs), copy(c.l), copy(c.u))
end


"""
For convenience, getindex of a Chebyshev polynomial just returns the coefficient with that index.
"""
function Base.getindex(c::ChebyshevPolynomial, i)
    return getindex(c.coeffs, i)
end


# TODO: Why does this not work for cheby[1:2] .= 5 ???
function Base.setindex!(c::ChebyshevPolynomial, value, i::Int)
    return ChebyshevPolynomial(setindex!(c.coeffs, value, i), c.l, c.u)
end


Base.iterate(c::ChebyshevPolynomial) = iterate(c.coeffs)
Base.iterate(c::ChebyshevPolynomial, s) = iterate(c.coeffs, s)


function dpoly(c::ChebyshevPolynomial)
    dcoeffs = chebyshev_derivative(c.coeffs, c.l, c.u)
    return ChebyshevPolynomial(dcoeffs, c.l, c.u)
end


function real_roots(c::ChebyshevPolynomial)
    chebyshev_roots(c.coeffs, c.l, c.u)
end


"""
Returns function evaluating a polynomial in Chebyshev basis.
"""
function make_eval_poly(c::ChebyshevPolynomial)
    p = x -> clenshaw_chebyshev(c.coeffs, x, c.l, c.u)
end