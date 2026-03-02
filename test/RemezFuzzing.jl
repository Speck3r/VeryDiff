
using VeryDiff

function test_poly_approx(ps, l, u; n_test=1000)
    poly = x -> sum(ps[k]*x^(k-1) for k in 1:length(ps))
    α, β, ϵ = VeryDiff.approx_polynomial_lin(ps, l, u, cheby=false)

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


function test_poly_approx_cheby(ps, l, u; n_test=1000)
    poly = VeryDiff.make_eval_chebyshev(ps, l, u)
    α, β, ϵ = VeryDiff.approx_polynomial_lin(ps, l, u, l, u, max_iter=20, cheby=true)
    flin = VeryDiff.make_eval_chebyshev([β, α], l, u)


    for i in 1:n_test
        x = l + (u - l)*rand()

        y_l = flin(x) - ϵ
        y_u = flin(x) + ϵ

        l_error = (poly(x) < y_l)
        u_error = (poly(x) > y_u)

        (l_error || u_error) && print("p(x) = ", poly(x), " but ")
        l_error && print("y_l = ", y_l, " ")
        u_error && print("y_u = ", y_u, " ")
        (l_error || u_error) && print("\n")
        (l_error || u_error) && println("\tps = ", ps)
    end
end


function test_relu_approx(l, u, degree; n_test=1000)
    f = x -> max(0, x)
    ps, ϵ = VeryDiff.approx_relu_poly(l, u, degree, cheby=false)
    poly = x -> sum(ps[k]*x^(k-1) for k in 1:length(ps))

    for i in 1:n_test
        x = l + (u - l)*rand()

        y_l = poly(x) - ϵ
        y_u = poly(x) + ϵ

        l_error = (poly(x) < y_l)
        u_error = (poly(x) > y_u)

        (l_error || u_error) && print("ReLU(x) = ", f(x), " but ")
        l_error && print("y_l = ", y_l, " ")
        u_error && print("y_u = ", y_u, " ")
        (l_error || u_error) && print("\n")
        (l_error || u_error) && println("\tps = ", ps)
    end
end


function test_relu_approx_cheby(l, u, degree; n_test=1000)
    f = x -> max(0, x)
    ps, ϵ = VeryDiff.approx_relu_poly(l, u, degree, cheby=true)
    poly = VeryDiff.make_eval_chebyshev(ps, l, u)

    for i in 1:n_test
        x = l + (u - l)*rand()

        y_l = poly(x) - ϵ
        y_u = poly(x) + ϵ

        l_error = (poly(x) < y_l)
        u_error = (poly(x) > y_u)

        (l_error || u_error) && print("ReLU(x) = ", f(x), " but ")
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


function test_poly_approx_cheby_random(;n_test=100, n_test_input=100)
    for i in 1:n_test
        degree = rand(2:6)
        ps = randn(degree+1)
        l = randn()
        u = l + abs(randn())
        test_poly_approx_cheby(ps, l, u, n_test=n_test_input)
    end
end


function test_relu_approx_random(;n_test=100, n_test_input=100, scale=1., cheby=true)
    for i in 1:n_test
        degree = rand(2:6)
        l = scale * randn()
        u = l + abs(scale * randn())
        if cheby
            test_relu_approx_cheby(l, u, degree, n_test=n_test_input)
        else
            test_relu_approx(l, u, degree, n_test=n_test_input)
        end
    end
end


test_poly_approx_random(n_test=1000, n_test_input=100)
test_poly_approx_cheby_random(n_test=1000, n_test_input=100)
test_relu_approx_random(n_test=1000, n_test_input=100, cheby=false)
test_relu_approx_random(n_test=1000, n_test_input=100, scale=1000, cheby=false)
test_relu_approx_random(n_test=1000, n_test_input=100, cheby=true)
test_relu_approx_random(n_test=1000, n_test_input=100, scale=1000, cheby=true)