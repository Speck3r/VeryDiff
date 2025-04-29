
using VeryDiff, Plots


l, u = -10000., 10000.
degree = 1
f = x -> max(0, x)
fc = VeryDiff.make_eval_chebyshev(f, l, u, degree)

cs, ϵ = VeryDiff.approx_relu_poly(l, u, degree, max_iter=1, cheby=true)
cp = VeryDiff.ChebyshevPolynomial(cs, l, u)
fr = VeryDiff.make_eval_poly(cp)

xs = range(l, u, 200)
plot(xs, f.(xs), label="ReLU")
plot!(xs, fc.(xs), label="cheby")
plot!(xs, fr.(xs), label="poly")


errs_cheby = []
errs_remez = []
for degree in 1:200
    cs, ϵ = VeryDiff.approx_relu_poly(l, u, degree, max_iter=1, cheby=true)
    push!(errs_cheby, ϵ)

    cs, ϵ = VeryDiff.approx_relu_poly(l, u, degree, max_iter=20, cheby=true)
    push!(errs_remez, ϵ)
end

plot(1:200, errs_cheby, yscale=:log, label="approx error")
plot!(1:200, errs_remez, label="remez error")

plot(50:200, errs_cheby[50:200], label="approx error")
plot!(50:200, errs_remez[50:200], label="remez error")

