
using VeryDiff, LinearAlgebra


l = -4.
u = 5.
degree = 200
cs, err = VeryDiff.approx_relu_poly(l, u, degree, verbosity=1)

α, β, ϵ = VeryDiff.approx_polynomial_lin(cs, l, u, l, u, verbosity=1)
y₀ = VeryDiff.clenshaw_chebyshev([β, α], 0., l, u)
y₁ = VeryDiff.clenshaw_chebyshev([β, α], 1., l, u)

a = (y₁ - y₀)

u / (u - l)


# equivalence verification is now slower, with larger REMEZ_ITERS, because we have of course more iterations involving expensive root finding.
# possible remedies:
#   - faster root finding
#   - don't use Remez algorithm: We already know that the polynomials are approximately ReLUs, so we just use the slope for the ReLU relaxation and only shift up or down to get 
#                                a valid relaxation for the polynomial