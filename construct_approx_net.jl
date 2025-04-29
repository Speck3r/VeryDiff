using VeryDiff, LinearAlgebra, VNNLib, DoubleFloats, Plots
import VeryDiff: approximate_polynomial_iterative, approximate_polynomial



function interleaved_approximate_polynomial(net, input_set, degree; verbosity=0, cheby=true, max_iter=20)
    prop_state = PropState(true)
    layers_poly = []
    ẑ = input_set
    for i in 1:length(net.layers)
        bounds_layer = zono_bounds(ẑ)
        layer = net.layers[i]
        layer_poly = approximate_polynomial(layer, bounds_layer, degree, cheby=cheby, verbosity=verbosity, max_iter=max_iter)
        push!(layers_poly, layer_poly)

        ẑ = layer_poly(ẑ, prop_state)

        verbosity > 0 && println("--- layer $i ---")
        verbosity > 0 && println("lower = ", bounds_layer[:,1][1:min(size(bounds_layer, 1), 5)])
        verbosity > 0 && println("upper = ", bounds_layer[:,2][1:min(size(bounds_layer, 1), 5)])
    end

    return Network(layers_poly)   
end





function get_empirical_bounds(net::Network, z::Zonotope; n_inputs=1000)
    bounds = []

    for i in 1:n_inputs
        x = VeryDiff.random_point(z)
        ys = VeryDiff.intermediate_activations(net, x)

        if length(bounds) == 0
            bounds = [[y y] for y in ys]
        else
            for j in 1:length(bounds)
                bounds[j][:,1] .= min.(bounds[j][:,1], ys[j])
                bounds[j][:,2] .= max.(bounds[j][:,2], ys[j])
            end
        end
    end

    return bounds 
end


function widen_bounds(bnds::AbstractVector, factor::N) where N<:Number
    bnd_new = deepcopy(bnds)
    for i in 1:length(bnds) 
        bnd = bnds[i]
        radius = 0.5 .* (bnd[:,2] .- bnd[:,1])
        center = 0.5 .* (bnd[:,1]  .+ bnd[:,2])

        bnd_new[i][:,1] .= center .- factor .* radius
        bnd_new[i][:,2] .= center .+ factor .* radius
    end 

    return bnd_new
end



## Example 

model_file = "./test/examples/networks/mnist-net_256x4.onnx"
model_file = "./test/examples/networks/mnist_256x4_2e5.onnx"
poly_dir = "./test/examples/poly_coeffs/models/mnist_fc/poly_2/"

z = Zonotope(I(784) .* 0.05, zeros(784), Float64.(I(784)));

nn = VNNLib.load_network(model_file);

test_set = [VeryDiff.random_point(z) for _ in 1:10000]
ys = [nn(x) for x in test_set]
Y  = hcat(ys...)';

t1 = @elapsed nn_poly = approximate_polynomial_iterative(nn, z, 100, verbosity=1, cheby=true);
t2 = @elapsed nn_poly2 = interleaved_approximate_polynomial(nn, z, 100, verbosity=1, cheby=true);

bnds_relu = VeryDiff.get_zono_bounds(nn, z)
bnds_poly = VeryDiff.get_zono_bounds(nn_poly, z);

z = Zonotope(I(784) .* 0.05 .* one(Double64), zeros(Double64, 784), Double64.(I(784)))
nn = VNNLib.Network([(typeof(l)<:VNNLib.Dense ? VNNLib.Dense(Double64.(l.W), Double64.(l.b)) : l) for l in nn.layers])
nn_poly = approximate_polynomial_iterative(nn, z, 100, verbosity=1, cheby=true);



println("## Zono bounds")
maes_zono = []
for degree in 100:25:300
    nn_poly = approximate_polynomial_iterative(nn, z, degree, verbosity=1, cheby=true)
    ys_poly = [nn_poly(x) for x in test_set]
    Y_poly = hcat(ys_poly...)'

    mse = sum(sum((Y .- Y_poly).^2, dims=2)) / size(Y, 1)
    mae = sum(maximum(abs.(Y .- Y_poly), dims=2)) / size(Y, 1)
    println("degree = ", degree, " - mse = ", mse, " - mae = ", mae)
    push!(maes_zono, mae)
end


println("## Zono bounds + Chebyshev only (no Remez)")
#maes_cheby_zono = []
for degree in 102:2:200
    nn_poly = approximate_polynomial_iterative(nn, z, degree, verbosity=0, cheby=true, max_iter=1)
    ys_poly = [nn_poly(x) for x in test_set]
    Y_poly = hcat(ys_poly...)'

    mse = sum(sum((Y .- Y_poly).^2, dims=2)) / size(Y, 1)
    mae = sum(maximum(abs.(Y .- Y_poly), dims=2)) / size(Y, 1)
    println("degree = ", degree, " - mse = ", mse, " - mae = ", mae)
    push!(maes_cheby_zono, mae)
end

bnds10k = get_empirical_bounds(nn, z, n_inputs=10000)
println("\n## Sampled bounds")
maes_sampled = []
for degree in 1:2:100
    nn_poly = approximate_polynomial(nn, bnds10k, degree)
    ys_poly = [nn_poly(x) for x in test_set]
    Y_poly = hcat(ys_poly...)'

    mse = sum(sum((Y .- Y_poly).^2, dims=2)) / size(Y, 1)
    mae = sum(maximum(abs.(Y .- Y_poly), dims=2)) / size(Y, 1)
    println("degree = ", degree, " - mse = ", mse, " - mae = ", mae)
    push!(maes_sampled, mae)
end

println("\n## Widened bounds 2")
wbnds = widen_bounds(bnds10k, 2)
maes_widen2 = []
for degree in 1:2:100
    nn_poly = approximate_polynomial(nn, wbnds, degree)
    ys_poly = [nn_poly(x) for x in test_set]
    Y_poly = hcat(ys_poly...)'

    mse = sum(sum((Y .- Y_poly).^2, dims=2)) / size(Y, 1)
    mae = sum(maximum(abs.(Y .- Y_poly), dims=2)) / size(Y, 1)
    println("degree = ", degree, " - mse = ", mse, " - mae = ", mae)
    push!(maes_widen2, mae)
end

println("\n## Widened bounds 4")
wbnds = widen_bounds(bnds10k, 4)
maes_widen4 = []
for degree in 1:2:100
    nn_poly = approximate_polynomial(nn, wbnds, degree, cheby=true)
    ys_poly = [nn_poly(x) for x in test_set]
    Y_poly = hcat(ys_poly...)'

    mse = sum(sum((Y .- Y_poly).^2, dims=2)) / size(Y, 1)
    mae = sum(maximum(abs.(Y .- Y_poly), dims=2)) / size(Y, 1)
    println("degree = ", degree, " - mse = ", mse, " - mae = ", mae)
    push!(maes_widen4, mae)
end


plot(1:2:100, maes_zono, yaxis=:log, label="zono", xlabel="approximation degree", ylabel="MAE")
plot!((1:2:100)[1:7], maes_sampled[1:7], label="sampled", linestyle=:dash)
plot!(1:2:100, maes_widen2, label="widen2")
plot!(1:2:100, maes_widen4, label="widen4")

plot(1:2:100, maes_zono, yaxis=:log, label="zono", xlabel="approximation degree", ylabel="MAE")
plot!(1:2:100, maes_widen2, label="widen2")
plot!(1:2:100, maes_widen4, label="widen4")








prop_state = PropState(true)

ẑ = nn(z, prop_state)
bounds_relu = zono_bounds(ẑ)

for degree in 1:10
    nn_poly = approximate_polynomial_iterative(nn, z, degree)

    ẑ_poly = nn_poly(z, prop_state)
    bounds_poly = zono_bounds(ẑ_poly)

    # propagate differential zonotope through the difference network
    nn_diff = GeminiNetwork(nn_poly, nn);
    ∂z = Zonotope(zeros(784, 784), zeros(784), nothing)
    zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)
    ẑΔ = nn_diff(zΔ, PropState(true))

    bounds_diff = zono_bounds(ẑΔ.∂Z)

    println("ReLU:\tupper = ", bounds_relu[:,2])
    println("Poly ($degree):\tupper = ", bounds_poly[:,2])
    println("Diff ($degree):\tupper = ", bounds_diff[:,2])
end





# debugging
nn_poly = approximate_polynomial(nn, wbnds, 30, cheby=false);
nn_diff = GeminiNetwork(nn_poly, nn);

∂z = Zonotope(zeros(784, 784), zeros(784), nothing)
zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)
ẑΔ = nn_diff(zΔ, PropState(true), wbnds, nothing)

bounds_diff = zono_bounds(ẑΔ.∂Z)


l = -98.64690179764429
u =  42.442529632602515
degree = 100
p, err = VeryDiff.approx_relu_poly(l, u, degree, verbosity=1, max_iter=20, tol=1e-13, plotting=false)
fp = VeryDiff.make_eval_chebyshev(p, l, u)

p̂ = VeryDiff.chebyshev_coefficients(fp, l, u, degree)
fp̂ = VeryDiff.make_eval_chebyshev(p̂, l, u)
errfun = (p, l, u) -> VeryDiff.poly_error_cheby(p̂, p, l, u)


δpoly = zeros(max(length(p), length(p̂)))
δpoly[1:length(p̂)] .+= p̂
δpoly[1:length(p)] .-= p
fδ = VeryDiff.make_eval_chebyshev(δpoly, l, u)
dδpoly = VeryDiff.chebyshev_derivative(δpoly, l, u)
fdδ = VeryDiff.make_eval_chebyshev(dδpoly, l, u)
xr = [x for x in VeryDiff.chebyshev_roots(dδpoly, l, u) if (l <= x) && (x <= u)]
xr = [xr; [l, u]]


α, β, ϵ = VeryDiff.approx_polynomial_lin(p, l, u, l, u, max_iter=20, tol=1e-13, verbosity=1)

cs1, err = VeryDiff.remez(fp, errfun, VeryDiff.poly_norm, l, u, 1, verbosity=2)
flin = VeryDiff.make_eval_chebyshev(cs1, l, u)
errfun(cs1, l, u)

plot(xs, fp.(xs), label="p(x)", framestyle=:origin)
plot!(xs, flin.(xs), label="lin")
plot!(xs, α .* xs .+ β, label="lin mono")


cs1 = VeryDiff.chebyshev_coefficients(fp, l, u, 1)



xs = range(l, u, 500)
plot(xs, fp.(xs), label="p(x)")
plot!(xs, fp̂.(xs), label="p̂(x)")

plot(xs, fδ, label="diff")
plot!(xs, fdδ.(xs), label="d fδ/ dx")
scatter!(xr, fδ.(xr), label="err")


plot(xs, max.(0, xs), label="relu", framestyle=:origin)
plot!(xs, fp.(xs), label="p(x)")
plot!(xs, α .* xs .+ β .+ ϵ, label="lin")
plot!(xs, fp.(xs) .- (α .* xs .+ β), label="p(x) - lin")

net_partial = VNNLib.Network(nn_poly.layers[1:idx-1])
VeryDiff.get_zono_bounds(net_partial, z)[end]





# NaN error
l = -6074.90561122097
u = 3659.866848800538
degree = 300
# ONLY works up to degree 42
# for 43 the error **increases** during Remez iterations
# for 44 we get NaN
VeryDiff.approx_relu_poly(l, u, degree, verbosity=1, max_iter=20, tol=1e-13, plotting=false)
VeryDiff.approx_relu_poly(l, u, 43, verbosity=1, cheby=false)


ld = DoubleFloat(l)
ud = DoubleFloat(u)
VeryDiff.approx_relu_poly(ld, ud, degree, verbosity=1, max_iter=20, tol=1e-13)

l = -6.
u = 4.
errs = []
for d in 1:400
    _, err = VeryDiff.approx_relu_poly(l, u, d, verbosity=0, max_iter=20, tol=1e-14)
    push!(errs, err)
end

C = abs(0.5*u*l/(u-l))

plot(errs, label="float64", yaxis=:log, xlabel="degree", ylabel="error")
plot!(C./(1:400), label="bound")


# convert network to DoubleFloat 
function net2dtype(nn, dtype=Double64)
    layers_dtype = []
    for l in nn.layers 
        if typeof(l) <: VNNLib.Dense 
            layer_dtype = VNNLib.Dense(dtype.(l.W), dtype.(l.b))
        elseif typeof(l) <: VeryDiff.MonomialPoly
            layer_dtype = VeryDiff.MonomialPoly(dtype.(l.coeffs))
        elseif typeof(l) <: VeryDiff.ChebyshevPoly
            layer_dtype = VeryDiff.ChebyshevPoly(dtype.(l.coeffs), dtype.(l.l), dtype.(l.u))
        else
            layer_dtype = l
        end

        push!(layers_dtype, layer_dtype)
    end

    return VNNLib.Network(layers_dtype)   
end

nn_double = net2dtype(nn, Double64);
nn_poly_double = approximate_polynomial_iterative(nn, z, 100, verbosity=1, cheby=true);
bnds_poly_double = VeryDiff.get_zono_bounds(nn_poly_double, z);



function bary_interp(x, w, xs, ys)
    terms = @. w / (x - xs)
    if any(isinf.(terms))     # there was division by zero
        # return the node's data value
        idx = findfirst(x.==xs)
        f = ys[idx]
    else
        f = sum(ys.*terms) / sum(terms)
    end
end


f = x -> max.(0, x)
x_cur = VeryDiff.chebyshev_points(degree+1, l, u)
x_curs = [x_cur]
sigma = ones(degree + 2)
sigma[2:2:end] .= -1.
for i in 1:10
    f_cur = f.(x_cur)
    w = VeryDiff.baryweights_chebfun(x_cur)
    h = (w' * f_cur) / (w' * sigma)
    p_cur = (f_cur .- h .* sigma)
    
    baryfun = x -> VeryDiff.barycentric_interpolation(x, p_cur, x_cur, w)
    p = VeryDiff.chebyshev_coefficients_vec(baryfun, l, u, degree)

    x_next, ϵ_max = VeryDiff.update_points(x_cur, h, p, VeryDiff.relu_error_cheby, l, u)
    x_cur = x_next
    push!(x_curs, x_cur)
end

x_cur = x_curs[end]
f_cur = f.(x_cur)
w = VeryDiff.baryweights_chebfun(x_cur)
h = (w' * f_cur) / (w' * sigma)
p_cur = (f_cur .- h .* sigma)
baryfun = x -> VeryDiff.barycentric_interpolation(x, p_cur, x_cur, w)

p = VeryDiff.chebyshev_coefficients_vec(baryfun, l, u, degree)
fp_cheb = VeryDiff.make_eval_chebyshev(p, l, u)

baryfun2 = x -> bary_interp(x, w, x_cur, p_cur)

x_err, y_err = VeryDiff.relu_error_cheby(p, l, u)
abs_err = abs.(y_err)
xr = [x_err[abs_err .> h]; x_cur]
yr = y_err[abs_err .> h]
perm = sortperm(xr)
xr_sorted = xr[perm]
err = [yr; sigma .* h][perm]
not_rep = [xr_sorted[2:end] .- xr_sorted[1:end-1] .!= 0; true]
xr_sorted_nr = xr_sorted[not_rep]
err_nr = err[not_rep]

x_next = [xr_sorted_nr[1]]
err_next = [err_nr[1]]
for i in 2:length(xr_sorted_nr)
    if (sign(err_nr[i]) == sign(err_next[end])) && (abs(err_nr[i]) > abs(err_next[end]))
        x_next[end] = xr_sorted_nr[i]
        err_next[end] = err_nr[i]
    elseif sign(err_nr[i]) != sign(err_next[end])
        push!(x_next, xr_sorted_nr[i])
        push!(err_next, err_nr[i])
    end
end

idx = argmax(abs.(err_next))
max_err = abs(err_next[idx])

d = max(idx - (degree + 2-1), 1)
x_next[d:d+(degree+2)-1]


xs = range(l, u, 500)
plot(xs, f.(xs), label="ReLU")
scatter!(x_cur, p_cur, label="px", markersize=1)
plot!(xs, baryfun(xs), label="bary")
plot!(xs, baryfun2.(xs), label="bary2")
plot!(xs, fp_cheb.(xs), label="cheby")
scatter!(x_err, f.(x_err), markersize=1, label="err")


# they are the same
maximum(abs.(baryfun(xs) .- baryfun2.(xs)))











# barycentric_weights seems to return all NaN 
# the numbers in the barycentric weights are all VERY close to zero, so dividing by them results in NaN 
# can we do some scaling???
x_cur = VeryDiff.chebyshev_points(degree+1, l, u)
VeryDiff.barycentric_weights(x_cur)

C = 4/(u - l)
n = length(x_cur)
ws = ones(n)
for j in 1:n 
    sign_prod = 1
    log_sum = 0.
    for v in 1:n 
        if v != j 
            sign_prod *= sign(x_cur[j] - x_cur[v])
            log_sum += log(abs(x_cur[j] - x_cur[v]))
        end
    end

    ws[j] = sign_prod / exp(n * log(1/C) + log_sum)
end




cs = VeryDiff.chebyshev_coefficients(x -> max(0, x), l, u, degree)
fp_cheby = x -> VeryDiff.clenshaw_chebyshev(cs, x, l, u)

xs = range(l, u, 300)
plot(xs, max.(0, xs), label="ReLU", framestyle=:origin)
plot!(xs, fp_cheby.(xs), label="p(x)")

dcs = VeryDiff.chebyshev_derivative(cs, l, u)
dp_cheby = x -> VeryDiff.clenshaw_chebyshev(dcs, x, l, u)
plot!(xs, dp_cheby.(xs), label="p'(x)")

findiff = x -> (fp_cheby(x + 1e-3) - fp_cheby(x)) / 1e-3
err = maximum(abs.(findiff.(xs[2:end-1]) .- dp_cheby.(xs[2:end-1])))
@show err
plot!(xs[2:end-1], findiff.(xs[2:end-1]), label="findiff")

errfun = x -> max(0, x) - fp_cheby(x)
cx = VeryDiff.chebyshev_coefficients(x -> x, l, u, 1)
dx = VeryDiff.chebyshev_derivative(cx, l, u)

cp = VeryDiff.chebyshev_derivative(.-cs, l, u)
f_cp = x -> VeryDiff.clenshaw_chebyshev(cp, x, l, u)
xs_zero = VeryDiff.chebyshev_roots(cp, l, u)

cp_pos = copy(cp)
cp_pos[1] += 1.
f_cppos = x -> VeryDiff.clenshaw_chebyshev(cp_pos, x, l, u)
xs_one = VeryDiff.chebyshev_roots(cp_pos, l, u)

plot(xs, f_cp.(xs), label="-p'(x)", framestyle=:origin)
scatter!(xs_zero, f_cp.(xs_zero), label="roots neg")
plot!(xs, f_cppos.(xs), label="1 - p'(x)")
scatter!(xs_one, f_cppos.(xs_one), label="roots pos")

x_err, y_err = VeryDiff.relu_error_cheby(cs, l, u)

sigma = ones(degree+2)
sigma[2:2:end] .= -1
x_cur = VeryDiff.chebyshev_points(degree+1, l, u)
f_cur = max.(0, x_cur)
w = VeryDiff.barycentric_weights(x_cur)
h = (w' * f_cur) / (w' * sigma)

p_cur = (f_cur .- h .* sigma)

bary = x -> VeryDiff.barycentric_interpolation(x, p_cur, x_cur, w)[1]
xs = range(l, u, 300)
plot(xs, max.(0, xs), label="ReLU", framestyle=:origin)
plot!(xs, fp_cheby.(xs), label="p(x)")
plot!(xs, bary.(xs), label="bary(x)")

err = maximum(abs.(fp_cheby.(xs) .- bary.(xs)))


# our barycentric weights are only different because of different scaling for [0,1] vs [l,u]
function baryweights(t)
    n = length(t)-1
    C = (t[n+1]-t[1]) / 4           # scaling factor to ensure stability
    tc = t/C
    # Adding one node at a time, compute inverses of the weights.
    ω = ones(n+1)
    for m in 0:n-1
        d = tc[1:m+1] .- tc[m+2]    # vector of node differences
        @. ω[1:m+1] *= d            # update previous
        ω[m+2] = prod( -d )         # compute the new one
    end
    w = 1 ./ ω 
end


function baryweights_chebfun(xs)
    n = length(xs)
    C = 4/(maximum(xs) - minimum(xs))
    w = ones(n)
    for j = 1:n 
        v = C*(xs[j] .- xs)
        v[j] = 1.
        vv = exp(sum(log.(abs.(v))))
        w[j] = 1/(prod(sign.(v))*vv)
    end 

    return w ./ maximum(abs.(w))
end

ŵ = baryweights(x_cur)

c_bary = VeryDiff.chebyshev_coefficients_vec(x -> VeryDiff.barycentric_interpolation(x, p_cur, x_cur, w), l, u, degree)
fc_bary = VeryDiff.make_eval_chebyshev(c_bary, l, u)

cp_bary = VeryDiff.chebyshev_derivative(.-c_bary, l, u)
fcp_bary = x -> VeryDiff.clenshaw_chebyshev(cp_bary, x, l, u)
xs_zero = VeryDiff.chebyshev_roots(cp_bary, l, u)


plot(xs, fc_bary.(xs), label="bary", framestyle=:origin)










prop_state = VeryDiff.PropState(true)
layers_poly = []
push!(layers_poly, nn.layers[1])  # start with linear layer already in the list
bound = nothing 

layer2 = nn.layers[2]
net_partial = Network(layers_poly)

ẑ = net_partial(z, prop_state)
bnds = zono_bounds(ẑ)

layer2_poly = approximate_polynomial(layer2, bnds, 20, cheby=true)
push!(layers_poly, layer2_poly);

push!(layers_poly, nn.layers[3])  # another linear layer
layer4 = nn.layers[4]
net_partial = Network(layers_poly)
ẑ = net_partial(z, prop_state)
bnds = zono_bounds(ẑ)

layer4_poly = approximate_polynomial(layer2, bnds, 20, cheby=true)


idx = 1
fp = x -> VeryDiff.clenshaw_chebyshev(layer4_poly.coeffs[idx,:], x, layer4_poly.l[idx], layer4_poly.u[idx])
xs = range(bnds[idx,1], bnds[idx,2], 300)
plot(xs, max.(0, xs), label="relu", framestyle=:origin)
plot!(xs, fp.(xs), label="p")


α, β, ϵ = VeryDiff.approx_polynomial_lin(layer2_poly.coeffs[idx,:], bnds[idx,1], bnds[idx,2], bnds[idx,1], bnds[idx,2]);
fp_lin = x -> VeryDiff.clenshaw_chebyshev([β, α], x, bnds[idx,1], bnds[idx,2])
plot!(xs, fp_lin.(xs), label="lin")

α, β, γ = VeryDiff.get_linear_relaxation(layer2_poly, bnds[:,1], bnds[:,2]);

plot!(xs, α[idx] .* xs .+ β[idx], label="lin_mono")
plot!(xs, α[idx] .* xs .+ β[idx] .+ γ[idx], label="up_mono", linestyle=:dash)
plot!(xs, α[idx] .* xs .+ β[idx] .- γ[idx], label="lo_mono", linestyle=:dash)







cs, err = VeryDiff.approx_relu_poly(bnds[idx,1], bnds[idx,2], 15, max_iter=5, cheby=true)
fp = x -> VeryDiff.clenshaw_chebyshev(cs, x, bnds[idx,1], bnds[idx,2])
plot(xs, max.(0, xs), label="relu", framestyle=:origin)
plot!(xs, fp.(xs), label="p")

cs_norm = VeryDiff.normalize_chebyshev(cs, bnds[idx,1], bnds[idx,2])
fp_norm = x -> VeryDiff.clenshaw_chebyshev(cs_norm, x);

plot!(xs, fp_norm.(xs), label="p_norm")


xs = range(-6, 6, 300)
#plot(xs, fp.(xs), label="p")
#plot!(xs, fp_norm.(xs), label="p_norm")
plot(xs, fp.(xs) .- fp_norm.(xs), label="p - p_norm")