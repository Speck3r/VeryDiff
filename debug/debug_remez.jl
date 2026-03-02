
using VeryDiff, Plots, SpecialFunctions


gelu = x -> 0.5 * x * (1 + erf(x / sqrt(2)))
f_gelu_pp = VeryDiff.make_eval_gelu_piecewise_poly(VeryDiff.GELU_PP)

l, u = -7.615006613076782, -6.092005290461426
degree = 90
p, ϵ = VeryDiff.approx_gelu_poly(l, u, degree)

f = VeryDiff.make_eval_chebyshev(p, l, u)

xs_err, ys_err = VeryDiff.piecewise_poly_error(VeryDiff.GELU_PP, p, l, u);

xs = range(l, u, 500)
plot(xs, gelu.(xs), label="gelu")
plot!(xs, f_gelu_pp.(xs), label="gelu pw")
plot!(xs, f.(xs), label="poly")

#plot(xs, gelu.(xs) .- f.(xs), label="err gelu")
#plot!(xs, f_gelu_pp.(xs) .- f.(xs), label="err pw")
#scatter!(xs_err, ys_err, label="extrema")

plot(xs, f_gelu_pp.(xs) .- f.(xs), label="err pw")
scatter!(xs_err, ys_err, label="extrema")
