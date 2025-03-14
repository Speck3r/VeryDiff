
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