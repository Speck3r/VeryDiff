
using VeryDiff

function test_poly_approx(ps, l, u; n_test=1000)
    poly = x -> sum(ps[k]*x^(k-1) for k in 1:length(ps))
    α, β, ϵ = VeryDiff.approx_polynomial_lin(ps, l, u)

    for i in 1:n_test
        x = l + (u - l)*rand()

        y_l = α * x + β - ϵ
        y_u = α * x + β + ϵ

        l_error = (poly(x) < y_l)
        u_error = (poly(x) > y_u)

        (l_error || u_error) && print("p(x) = ", poly(x), " but ")
        l_error && print("y_l = ", y_l, " ")
        u_error && print("y_u = ", y_u, " ")
        (l_error || u_error) && print("\n")
        (l_error || u_error) && println("\tps = ", ps)
    end
end

function test_poly_approx_random(;n_test=100, n_test_input=100)
    for i in 1:n_test
        degree = rand(2:6)
        ps = randn(degree+1)
        l = randn()
        u = l + abs(randn())
        test_poly_approx(ps, l, u, n_test=n_test_input)
    end
end


test_poly_approx_random(n_test=1000, n_test_input=100)