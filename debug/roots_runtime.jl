
using VeryDiff, Plots


f = x -> max(0, x)
l = -2.
u = 4.

degree = 100
c = VeryDiff.chebyshev_coefficients(f, l, u, degree)
fc = VeryDiff.make_eval_chebyshev(c, l, u)

VeryDiff.real_roots(c)


degrees = 1:200
times = zeros(length(degrees))
trials = 200
for (i, degree) in enumerate(degrees)
    c = VeryDiff.chebyshev_coefficients(f, l, u, degree)
    fc = VeryDiff.make_eval_chebyshev(c, l, u)

    ts_trial = zeros(trials)
    for (i, trial) in enumerate(1:trials)
        t = @elapsed VeryDiff.real_roots(c)
        ts_trial[i] = t
    end
    ts_sorted = sort(ts_trial)
    t = iseven(length(ts_sorted)) ? 0.5 * (ts_sorted[Int(length(ts_sorted)//2)] + ts_sorted[Int(length(ts_sorted)//2 + 1)]) : ts_sorted[Int(ceil(length(ts_sorted)//2))]
    println("degree = ", degree, " -- time = ", t)
    times[i] = t
end

plot(1:200, times, label="real_roots() time", xlabel="degree", ylabel="median time")

plot(1:200, times, label="real_roots() time", xlabel="degree", ylabel="median time", yscale=:log)