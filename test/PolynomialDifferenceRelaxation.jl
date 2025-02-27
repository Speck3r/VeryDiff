
using VeryDiff

# test that version using monomial coefficients and version using Chebyshev polynomials 
# produce the same results
function test_monomial_vs_chebyshev_diff_relaxation(l, u, degree)
    # TODO: or just change the lx, ux, lΔ, uΔ in the find_good_poly_diff_approx() below.
    @assert (l <= -1) && (u >= 1) "make sure that the relaxation was fit at least to [-1, 1]"

    p = VeryDiff.chebyshev_approximation(x -> max(0, x), l, u, degree)
    fp = x -> sum(p[k]*x^(k-1) for k in 1:length(p))

    c = VeryDiff.chebyshev_coefficients(fp, l, u, degree)
    fc = VeryDiff.make_eval_chebyshev(c, l, u)

    #xs = range(l, u, 200)
    #plot(xs, fp.(xs), label="poly")
    #plot!(xs, fc.(xs), label="cheby")

    ap, bp, cp, ϵp = VeryDiff.find_good_poly_diff_approx(-1., 1., -1., 1., p)
    ac, bc, cc, ϵc = VeryDiff.find_good_poly_diff_approx(-1., 1., -1., 1., VeryDiff.ChebyshevPolynomial(c, l, u))

    @show (ap, bp, cp, ϵp)
    @show (ac, bc, cc, ϵc)
end