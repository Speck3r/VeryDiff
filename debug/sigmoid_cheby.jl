
# This script demonstrates the use of Chebyshev polynomials to approximate a function that is infinitely differentiable (in our example, the sigmoid function).
# We will also compare the approximation error of the sigmoid function with the ReLU function.
# Note that the approximation error for the sigmoid function is rapidly decreasing (compared to the very slow convergence for ReLU).


using VeryDiff, Plots

l = -5.
u = 5.
degree = 3
f = x -> 1 / (1 + exp(-x))
fc = VeryDiff.make_eval_chebyshev(f, l, u, degree)


xs = range(l, u, 200)
plot(xs, f.(xs), label="sigmoid")
plot!(xs, fc.(xs), label="cheby")


findiff(f; ϵ=1e-4) = x -> (f(x + ϵ) - f(x)) / ϵ

plot(xs, f.(xs), label="σ(x)")
plot!(xs, findiff(f).(xs), label="σ'(x)")
plot!(xs, findiff(findiff(f)).(xs), label="σ''(x)")
plot!(xs, findiff(findiff(findiff(f))).(xs), label="σ'''(x)")


function approx_diff(f1, f2, l, u; n_test=10000)
    xs = range(l, u, n_test)
    diff = maximum(abs.(f1.(xs) .- f2.(xs)))
end

diffs = []
for degree in 1:100
    fc = VeryDiff.make_eval_chebyshev(f, l, u, degree)
    diff = approx_diff(f, fc, l, u)
    push!(diffs, diff)
end


err_bound(n; ν=1, V=1) = 4*V / (π*ν*(n - ν)^ν)

V = abs(f(l) - f(u))

degrees = 1:100
plot(degrees, diffs, yscale=:log, label="approx error")
plot!(degrees, err_bound.(degrees, ν=1, V=V), label="ν=1")
plot!(degrees[2:end], err_bound.(degrees[2:end], ν=2, V=V), label="ν=2")
plot!(degrees[3:end], err_bound.(degrees[3:end], ν=3, V=V), label="ν=3")


degrees = 1:100
diffs_relu = []
diffs_sigmoid = []
for degree in degrees
    fc = VeryDiff.make_eval_chebyshev(f, l, u, degree)
    diff_sigmoid = approx_diff(f, fc, l, u)
    push!(diffs_sigmoid, diff_sigmoid)
    fc = VeryDiff.make_eval_chebyshev(x -> max(0, x), l, u, degree)
    diff_relu = approx_diff(x -> max(0, x), fc, l, u)
    push!(diffs_relu, diff_relu)
end

plot(degrees, diffs_sigmoid, yscale=:log, label="err sigmoid", xlabel="degree", ylabel="error")
plot!(degrees, diffs_relu, label="err relu")
