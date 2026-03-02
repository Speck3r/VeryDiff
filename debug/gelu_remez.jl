
using VeryDiff, SpecialFunctions, Plots, JLD2, LinearAlgebra


gelu = x -> 0.5 * x * (1 + erf(x / sqrt(2)))


l, u = -100., 100.
degree_gold = 100
gs = VeryDiff.chebyshev_coefficients(gelu, l, u, degree_gold)
f_g = VeryDiff.make_eval_chebyshev(gs, l, u)
g_err = x -> gelu(x) - f_g(x)

xs = range(l, u, 200)
plot(xs, gelu.(xs), label="GeLU", framestyle=:origin)
plot!(xs, f_g.(xs), label="gt poly")

plot(xs, g_err.(xs), label="gt error", framestyle=:origin)


degree = 5
f_cheby = VeryDiff.make_eval_chebyshev(gelu, l, u, degree)
p_approx, ϵ = VeryDiff.remez(f_g, (p, l, u) -> VeryDiff.poly_error_cheby(gs, p, l, u), VeryDiff.poly_norm, l, u, degree, verbosity=1, max_iter=20)

f_cheby = VeryDiff.make_eval_chebyshev(gelu, l, u, degree)
f_p = VeryDiff.make_eval_chebyshev(p_approx, l, u);

plot(xs, g_err.(xs), label="gt error", framestyle=:origin)
plot!(xs, gelu.(xs) .- f_cheby.(xs), label="Cheby 5 error")
plot!(xs, gelu.(xs) .- f_p.(xs), label="Remez 5 error")


# TODO: can we assign a type to polys?
struct GeLUPiecewisePoly{N,VN,VVN}
    a::N   # linear segment in (-∞, a]
    b::N   # linear segment in [b, ∞)
    coeffs::VVN  # coefficients of polynomials
    ls::VN  # lower bounds for polynomial segments
    us::VN  # upper bounds for polynomial segments
    polys  # functions to evaluate the polynomials
end


function make_piecewise_poly(params)
    poly_coeffs = first.(params["polys"])
    ls = getindex.(params["polys"], 2)
    us = getindex.(params["polys"], 3)
    polys = [VeryDiff.make_eval_chebyshev(cs, l, u) for (cs, l, u) in zip(poly_coeffs, ls, us)]
    GeLUPiecewisePoly(params["linear_up_to"], params["linear_from_on"], poly_coeffs, ls, us, polys)
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
function piecewise_poly_error(pp::GeLUPiecewisePoly{N,VN,VVN}, p::AbstractVector{N}, l::N, u::N) where {N,VN,VVN}
    xs_all = Vector{N}()

    # executable function for p
    poly_candidate = VeryDiff.make_eval_chebyshev(p, l, u)
    degree = length(p) - 1

    if l < pp.a
        # we only need to execute the whole block if we intersect with the domain of the 0-approximation
        # error for the linear 0 approximation up until a 
        # so GeLU(x) - p(x) ≈ 0 - p(x) = -p(x)
        println("in branch l < pp.a!")
        ∇p = VeryDiff.chebyshev_derivative(.-p, l, u)
        xs_zero = VeryDiff.chebyshev_roots(∇p, l, u)
        @show xs_zero
        xs_zero = [x for x in xs_zero if (l <= x) && (x <= pp.a)]
        @show xs_zero
        push!(xs_all, xs_zero...)
    end

    # error for the polynomial pieces 
    # here GeLU(x) - p(x) ≈ q(x) - p(x)
    # so we can reuse the already defined polyonmial error 
    for (l̂, û, q) in zip(pp.ls, pp.us, pp.coeffs)
        # only need to care about this polynomial piece if the approximation intervals intersect,
        # i.e. [l̂, û] ∩ [l, u] ≠ ∅
        l_intersect = max(l̂, l)
        u_intersect = min(û, u)
        if l <= l_intersect && u_intersect <= u
            # convert p to the approximation domain of the polynomial piece
            ps = VeryDiff.chebyshev_coefficients(poly_candidate, l̂, û, degree)
            
            xs, ys = VeryDiff.poly_error_cheby(q, ps, l̂, û)
            xs = [x for x in xs if (l_intersect <= x && x <= u_intersect)]
            push!(xs_all, xs...)
        end
    end

    if pp.b < u 
        # only need to care, if we intersect with the domain of the identity approximation.
        # error for the linear identity approximation from b on
        # so GeLU(x) - p(x) ≈ x - p(x)
        # need Chebyshev coefficients of f(x) = x over [l,u]
        cx = VeryDiff.chebyshev_coefficients(x -> x, l, u, 1)
        dx = VeryDiff.chebyshev_derivative(cx, l, u)
        ∇p = VeryDiff.chebyshev_derivative(.-p, l, u)
        ∇p[1] += dx[1]
        xs_one = VeryDiff.chebyshev_roots(∇p, l, u)
        xs_one = [x for x in xs_one if (pp.b <= x) && (x <= u)]
        push!(xs_all, xs_one...)
    end

    f_g = make_eval_gelu_piecewise_poly(pp)
    ys_all = f_g.(xs_all) .- poly_candidate.(xs_all)

    xs_all, ys_all
end




# piecewise_params = load(joinpath(@__DIR__, "..", "..", "PolynomialEquivalenceNN", "gelu_piecewise_poly_sampled_degree_15.jld2"))["params"]
# gpp = make_piecewise_poly(piecewise_params)
f_gpp = VeryDiff.make_eval_gelu_piecewise_poly(VeryDiff.GELU_PP)


l, u = -5., 5.
xs = range(l, u, 200)
degree = 10;


plot(xs, f_gpp.(xs))
plot(xs, gelu.(xs) .- f_gpp.(xs))

VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false
VeryDiff.ROOTS_ALMOST_ZERO_TOL[] = 1e-15
p_cheby = VeryDiff.chebyshev_coefficients(gelu, l, u, degree)
p_approx_pw, ϵ = VeryDiff.remez(f_gpp, (p, l, u) -> VeryDiff.piecewise_poly_error(VeryDiff.GELU_PP, p, l, u), VeryDiff.poly_norm, l, u, degree, verbosity=1, max_iter=20)

f_cheby = VeryDiff.make_eval_chebyshev(p_cheby, l, u)
f_pw = VeryDiff.make_eval_chebyshev(p_approx_pw, l, u);


xs_err, ys_err = VeryDiff.piecewise_poly_error(VeryDiff.GELU_PP, p_cheby, l, u);

plot(xs, gelu.(xs) .- f_cheby.(xs), label="cheby", framestyle=:origin)
plot!(xs, gelu.(xs) .- f_pw.(xs), label="remez")
scatter!(xs_err, ys_err, label="extrema")



function eval_gelu_approx_radius(r, degree; max_iter=20)
    l, u = -r, r 
    xs = range(l, u, 1000)

    p_cheby = VeryDiff.chebyshev_coefficients(gelu, l, u, degree)
    p_approx_pw, ϵ = VeryDiff.remez(f_gpp, (p, l, u) -> VeryDiff.piecewise_poly_error(VeryDiff.GELU_PP, p, l, u), VeryDiff.poly_norm, l, u, degree, verbosity=0, max_iter=max_iter)
    f_cheby = VeryDiff.make_eval_chebyshev(p_cheby, l, u)
    f_pw = VeryDiff.make_eval_chebyshev(p_approx_pw, l, u);

    xs_err, ys_err = VeryDiff.piecewise_poly_error(VeryDiff.GELU_PP, p_cheby, l, u)

    sampled_err_remez = maximum(abs.(gelu.(xs) .- f_pw.(xs)))
    sampled_err_cheby = maximum(abs.(gelu.(xs) .- f_cheby.(xs)))
    # computed_err_remez= maximum(abs.(ys_err))
    computed_err_remez = ϵ

    return sampled_err_cheby, sampled_err_remez, computed_err_remez
end

degree = 50
radii = range(1, 100, 100)
sampled_errs_cheby = []
sampled_errs_remez = []
computed_errs_remez = []
for r in radii
    sampled_err_cheby, sampled_err_remez, computed_err_remez = eval_gelu_approx_radius(r, degree)
    push!(sampled_errs_cheby, sampled_err_cheby)
    push!(sampled_errs_remez, sampled_err_remez)
    push!(computed_errs_remez, computed_err_remez)
end

plot(radii, sampled_errs_cheby, label="cheby", yscale=:log10)
plot!(radii, sampled_errs_remez, label="remez sampled")
plot!(radii, computed_errs_remez, label="remez computed")



