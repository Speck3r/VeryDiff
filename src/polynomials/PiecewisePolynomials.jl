
struct GeLUPiecewisePoly{N,VN,VVN,FN}
    a::N   # linear segment in (-∞, a]
    b::N   # linear segment in [b, ∞)
    coeffs::VVN  # coefficients of polynomials
    ls::VN  # lower bounds for polynomial segments
    us::VN  # upper bounds for polynomial segments
    polys::FN  # functions to evaluate the polynomials
    ϵ::N  # maximum error between gelu and the polynomials
end


"""
    `load_piecewise_poly(path; no_error=false)`

Loads a piecewise polynomial stored in a .jld2 file given by the path.

If you don't care about the error between the piecewise polynomials and the function they approximate,
you can pass no_error=true.
"""
function load_piecewise_poly(path; no_error=false)
    params = load(path)["params"]
    poly_coeffs = first.(params["polys"])
    ls = getindex.(params["polys"], 2)
    us = getindex.(params["polys"], 3)

    if no_error
        ϵ = 0.
    else
        ϵ  = maximum(params["errors"])
    end

    polys = [make_eval_chebyshev(cs, l, u) for (cs, l, u) in zip(poly_coeffs, ls, us)]
    GeLUPiecewisePoly(params["linear_up_to"], params["linear_from_on"], poly_coeffs, ls, us, polys, ϵ)
end


function make_eval_gelu_piecewise_poly(pp::GeLUPiecewisePoly)
    x -> begin
        if x <= pp.a 
            return 0.
        elseif x >= pp.b 
            return x 
        else 
            for (l, u, p) in zip(pp.ls, pp.us, pp.polys)
                if (x < l) || (x > u)
                    continue
                end 

                return p(x)
            end
        end 
    end
end


"""
Approximates extrema of GeLU(x) - p(x) in Chebyshev representation over interval [l, u].

!!! We require p to have approximation domain [l, u] !!!

args:
    pp - piecewise polynomial approximation of GeLU for (-∞, ∞)
    p  - Chebyshev coefficients of the approximating polynomial in order [p₀, p₁, ...]
    l  - concrete lower bound of the approximation interval 
    u  - concrete upper bound of the approximation interval 

returns:
    xs - locations of the extrema 
    ys - approximated values for GeLU(x) - p(x) at the extrema

"""
function piecewise_poly_error(pp::GeLUPiecewisePoly{N,VN,VVN,FN}, p::AbstractVector{N}, l::N, u::N) where {N,VN,VVN,FN}
    # xs_all = Vector{N}()
    xs_all = [l, u]

    # executable function for p
    poly_candidate = make_eval_chebyshev(p, l, u)
    degree = length(p) - 1

    if l < pp.a
        # we only need to execute the whole block if we intersect with the domain of the 0-approximation
        # error for the linear 0 approximation up until a 
        # so GeLU(x) - p(x) ≈ 0 - p(x) = -p(x)
        ∇p = chebyshev_derivative(.-p, l, u)
        xs_zero = chebyshev_roots(∇p, l, u)
        xs_zero = [x for x in xs_zero if (l <= x) && (x <= pp.a)]
        push!(xs_all, xs_zero...)
    end

    # error for the polynomial pieces 
    # here GeLU(x) - p(x) ≈ q(x) - p(x)
    # so we can reuse the already defined polyonmial error 
    for (l̂, û, q, f_q) in zip(pp.ls, pp.us, pp.coeffs, pp.polys)
        # only need to care about this polynomial piece if the approximation intervals intersect,
        # i.e. [l̂, û] ∩ [l, u] ≠ ∅
        l_intersect = max(l̂, l)
        u_intersect = min(û, u)
        if l_intersect <= u_intersect
            if (l̂ == l_intersect) && (u_intersect == û)
                # if intersection spans whole approximation interval of the polynomial piece,
                # we can just convert the poly to the domain of the polynomial piece.
                # Everything will behave nicely since both polys are within their approximation domain.

                # convert p to the approximation domain of the polynomial piece
                ps = chebyshev_coefficients(poly_candidate, l̂, û, degree)
                
                xs, ys = poly_error_cheby(q, ps, l̂, û)
            else 
                # polynomials can behave nastily outside of their approximation domain.
                # We need to convert both of them to the intersection domain, where both behave nicely.

                # convert p and q to the intersection domain
                ps = chebyshev_coefficients(poly_candidate, l_intersect, u_intersect, degree)
                qs = chebyshev_coefficients(f_q, l_intersect, u_intersect, degree)

                xs, ys = poly_error_cheby(qs, ps, l_intersect, u_intersect)
            end

            xs = [x for x in xs if (l_intersect <= x && x <= u_intersect)]
            push!(xs_all, xs...)
        end
    end

    if pp.b < u 
        # only need to care, if we intersect with the domain of the identity approximation.
        # error for the linear identity approximation from b on
        # so GeLU(x) - p(x) ≈ x - p(x)
        # need Chebyshev coefficients of f(x) = x over [l,u]
        cx = chebyshev_coefficients(x -> x, l, u, 1)
        dx = chebyshev_derivative(cx, l, u)
        ∇p = chebyshev_derivative(.-p, l, u)
        ∇p[1] += dx[1]
        xs_one = chebyshev_roots(∇p, l, u)
        xs_one = [x for x in xs_one if (pp.b <= x) && (x <= u)]
        push!(xs_all, xs_one...)
    end

    f_g = make_eval_gelu_piecewise_poly(pp)
    ys_all = f_g.(xs_all) .- poly_candidate.(xs_all)

    xs_all, ys_all
end