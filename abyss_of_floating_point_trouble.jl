using VeryDiff, Plots


## Demonstration: Numerical accuracy of Chebyshev Interpolation and Clenshaw vs. Chebyshev via Vandermonde matrix and Monomial evaluation
f = x -> max(0, x)
l = -5
u = 5
degree = 100
kind = 2

fc = VeryDiff.chebyshev_approximation(f, l, u, degree, kind=kind)
fp = make_eval_poly(fc)

fc_cheb = VeryDiff.chebyshev_coefficients(f, l, u, degree, kind=kind)
fp_cheb = x -> VeryDiff.clenshaw_chebyshev(fc_cheb, x, l, u)

xs = range(l, u, 300)
plot(xs, f.(xs), label="relu", framestyle=:origin)
plot!(xs, fp.(xs), label="monomial")
plot!(xs, fp_cheb.(xs), label="Clenshaw")



## Demonstration: Reexpansion of Chebyshev polynomial outside of the approximation domain is usually a BAD idea!
#       If it was fit to a well-behaved function, the interpolant is well-behaved on the approximatoin domain, but
#       shoots off to ±∞ very quickly outside of that interval.
#       If you fit another Chebyshev interpolant to the approximation polynomial in the badly behaved region, it will 
#       also behave badly due to numerical error.
f = x -> max(0, x)
l = -5.
u = 5.
degree = 20
kind = 2

cs, err = VeryDiff.approx_relu_poly(l, u, degree, max_iter=5, cheby=true)
fp = x -> VeryDiff.clenshaw_chebyshev(cs, x, l, u)

cs_norm = VeryDiff.normalize_chebyshev(cs, l, u)
fp_norm = x -> VeryDiff.clenshaw_chebyshev(cs_norm, x);

xs = range(-1, 1, 300)
pl1 = plot(xs, max.(0, xs), label="relu", framestyle=:origin)
plot!(xs, fp.(xs), label="p")
plot!(xs, fp_norm.(xs), label="p_norm")

pl2 = plot(xs, fp.(xs) .- fp_norm.(xs), label="p - p_norm")


xs = range(l-5, u+5, 300)
pl3 = plot(xs, max.(0, xs), label="relu", framestyle=:origin)
plot!(xs, fp.(xs), label="p")
plot!(xs, fp_norm.(xs), label="p_norm")

pl4 = plot(xs, fp.(xs) .- fp_norm.(xs), label="p - p_norm")

plot(pl1, pl2, pl3, pl4, layout=(2,2))