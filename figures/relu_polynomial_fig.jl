
using VeryDiff, Plots


## Show that polynomial approximation outside of approximation domain explodes to ±∞
f = x -> max(0, x)
l = -5.
u = 5.
cs3, ϵ3 = VeryDiff.approx_relu_poly(l, u, 3, verbosity=1)
cs5, ϵ5 = VeryDiff.approx_relu_poly(l, u, 5, verbosity=1)
cs10, ϵ10 = VeryDiff.approx_relu_poly(l, u, 10, verbosity=1)

fc3 = VeryDiff.make_eval_chebyshev(cs3, l, u)
fc5 = VeryDiff.make_eval_chebyshev(cs5, l, u)
fc10 = VeryDiff.make_eval_chebyshev(cs10, l, u)

xs = range(l-1, u+1, 200)
plot(xs, f.(xs), label="relu", framestyle=:origin, xlabel="x", ylabel="y")
plot!(xs, fc3.(xs), label="degree 3")
plot!(xs, fc5.(xs), label="degree 5")
plot!(xs, fc10.(xs), label="degree 10")
vline!([l, u], label="bounds", color=:black, linestyle=:dash)

savefig("relu_poly_approx_out_of_bounds.pdf")


## Show that zonotope approximation of ReLU can produce worse bounds, even if the input set is smaller.
f = x -> max(0, x)
l = -5.
u = 5.

l1 = -2.
λ1 = u/(u-l)
β1 = -l .* λ1
λ2 = u/(u-l1)
β2 = -l1 .* λ2

xs = range(l, u, 200)
xs_tight = range(l1, u, 200)
plot(xs, f.(xs), label="relu", framestyle=:origin, xlabel="x", ylabel="y")
plot!(xs, λ1 .* xs .+ β1, label="ub", color=2)
plot!(xs, λ1 .* xs, label="lb", color=2)
plot!(xs, λ1 .* xs, fillrange=λ1 .* xs .+ β1, alpha=0.2, color=2, primary=false)
plot!(xs_tight, λ2 .* xs_tight .+ β2, label="ub tight", color=3)
plot!(xs_tight, λ2 .* xs_tight, label="lb tight", color=3)
plot!(xs_tight, λ2 .* xs_tight, fillrange=λ2 .* xs_tight .+ β2, alpha=0.2, color=3, primary=false)
vline!([l1], label="tighter l", color=:black, linestyle=:dash)

savefig(string(@__DIR__, "/relu_incomparable_relaxations.pdf"))

# We see that the relaxations are incomparable, because there are regions, where one is feasible and the other is not.
# TODO: find a ReLU NN where this actually leads to worse bounds for the smaller input set.