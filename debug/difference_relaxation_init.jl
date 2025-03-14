
using VeryDiff, Plots

# want to have p(x) - ReLU(y) = p(x) - ReLU(x - (x - y)) = p(x) - ReLU(x - Δ)
lx = -2.
ux = 2.
ly = -1. 
uy = 1.
lΔ = -3.
uΔ = 4.
degree = 30
c = VeryDiff.chebyshev_coefficients(x -> max(0, x), lx, ux, degree)
fc = VeryDiff.make_eval_chebyshev(c, lx, ux)
f = (x, Δ) -> fc(x) - max(0, x - Δ)
f_relu = (x, Δ) -> max(0, x) - max(0, x - Δ)

a, b = VeryDiff.poly_initial_slope_guess(lx, ux, ly, uy, lΔ, uΔ)
a, b, c, ϵ = VeryDiff.find_good_poly_diff_approx(lx, ux, lΔ, uΔ, VeryDiff.ChebyshevPolynomial(c, lx, ux), a, b[1])

xs = range(lx, ux, 100)
Δs = range(lΔ, uΔ, 100)
u_relax = (x, Δ) -> a*x + b*Δ + c + ϵ
surface(xs, Δs, f, label="diff", xlabel="x", ylabel="Δ")
# surface!(xs, Δs, f_relu, label="relu_diff", alpha=0.2, color=:blues)  # they look basically the same
surface!(xs, Δs, u_relax, label="u_relax", alpha=0.5)






