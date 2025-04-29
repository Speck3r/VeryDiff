

using VeryDiff, VNNLib, CSV, Plots, JLD2, Dates, LinearAlgebra



f_mnist = CSV.File(string(@__DIR__, "/data/mnist_train.csv"), header=false)
X_test = [Float64.([x for x in f_mnist[i]][2:end]) ./ 255 for i in 1:size(f_mnist, 1)]
y_test = [[x for x in f_mnist[i]][1] for i in 1:size(f_mnist, 1)]

X_mat = hcat(X_test...)'
x_min = vec(minimum(X_mat, dims=1))
x_max = vec(maximum(X_mat, dims=1))

z = Zonotope(x_min, x_max)
# z = Zonotope(zeros(784), ones(784))


res = load("./mnist_4x256_1e4_zono_polys_data_2025-03-27T17:39:57.751.jld2")
nets_l1_1e4 = res["nets_l1_1e4"];
maes_l1_1e4 = res["maes_l1_1e4"];

model_file = string(@__DIR__, "/networks/mnist_256x4_1e4.onnx")
nn = VNNLib.load_network(model_file)

degrees = (1:9) ∪ (10:10:200)
VeryDiff.OPTIM_ITERS[] = 0
prop_state = PropState(true)
∂bounds = []
for (i, nn_poly) in enumerate(nets_l1_1e4)
    # propagate differential zonotope through the difference network
    nn_diff = GeminiNetwork(nn_poly, nn);

    dom = VeryDiff.extract_approximation_domain(nn_poly)

    ∂z = Zonotope(zero(z.G), zero(z.c), nothing)
    zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)
    #ẑΔ = nn_diff(zΔ, PropState(true))
    ẑΔ = nn_diff(zΔ, PropState(true), dom, nothing)

    bounds_diff = zono_bounds(ẑΔ.∂Z)
    ∂bound = maximum(abs.(bounds_diff))
    println("\ndegree = ", degrees[i], " ∂bound = ", ∂bound)
    println("\tlbs = ", bounds_diff[:,1])
    println("\tubs = ", bounds_diff[:,2])
    println("")
    push!(∂bounds, ∂bound)
end


plot(degrees, ∂bounds, yscale=:log, xlabel="degree", ylabel="verified difference", marker=:diamond, label="verified diff", title="MNIST 4x256 1e-4", ylims=(0.9, maximum(∂bounds)+100))

jldsave(string("mnist_4x256_1e4_verified_difference_", now(), ".jld2"); ∂bounds)



## Equivalence on neighborhoods around the training data
VeryDiff.ALMOST_ZERO_LEADING_COEFF_WARNING[] = false
VeryDiff.USE_REWRITE_DIFF[] = true
# VeryDiff.REMEZ_ITERS[] = 10

n_test = 10
r = 0.1
∂bounds_sample = []
for (i, nn_poly) in enumerate(nets_l1_1e4)
    println("### degree = ", degrees[i])

    nn_diff = GeminiNetwork(nn_poly, nn);
    dom = VeryDiff.extract_approximation_domain(nn_poly)

    ∂bounds_net = []
    for (j, x₀) in enumerate(X_test[1:n_test])
        l = clamp.(x₀ .- r, 0, 1)
        u = clamp.(x₀ .+ r, 0, 1)

        z = Zonotope(l, u)
        ∂z = Zonotope(zero(z.G), zero(z.c), nothing)
        zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)

        zΔ = nn_diff(zΔ, PropState(true), dom, nothing)

        bounds_diff = zono_bounds(zΔ.∂Z)
        ∂bound = maximum(abs.(bounds_diff))
        push!(∂bounds_net, ∂bound)
        println("\tinput $j : ∂bound = ", ∂bound)
    end

    push!(∂bounds_sample, ∂bounds_net)
end

# save the bounds for the samples
jldsave(string("mnist_4x256_1e4_verified_difference_samples_", now(), ".jld2"); ∂bounds_sample)

∂bounds = load(string("./mnist_4x256_1e4_verified_difference_2025-04-01T15:28:49.184.jld2"))["∂bounds"]
p = plot(degrees, ∂bounds, label="verified error (all)", xlabel="degree", ylabel="verified error", yscale=:log, marker=:diamond, title="MNIST 4x256 1e-4 Verified Error")
for (i, bnds) in enumerate(∂bounds_sample)
    if i == 1
        scatter!(repeat([degrees[i]], n_test), bnds[11:end], label="verified error (samples)", color=2)
    else
        scatter!(repeat([degrees[i]], n_test), bnds[11:end], label="sample $i", primary=false, color=2)
    end
end
p



function verify_bounds_sample(X_test, nn, nn_poly, r, n_test)
    nn_diff = GeminiNetwork(nn_poly, nn);
    dom = VeryDiff.extract_approximation_domain(nn_poly)

    ∂bounds_net = []
    for (j, x₀) in enumerate(X_test[1:n_test])
        l = clamp.(x₀ .- r, 0, 1)
        u = clamp.(x₀ .+ r, 0, 1)

        z = Zonotope(l, u)
        ∂z = Zonotope(zero(z.G), zero(z.c), nothing)
        zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)

        t = @elapsed zΔ = nn_diff(zΔ, PropState(true), dom, nothing)

        bounds_diff = zono_bounds(zΔ.∂Z)
        ∂bound = maximum(abs.(bounds_diff))
        push!(∂bounds_net, ∂bound)
        println("\tinput $j : ∂bound = ", ∂bound, " (", t, "s)")
    end

    return ∂bounds_net
end


function verify_delta_top1(nn, nn_poly, delta)
    property_check = get_top1_property(naive=false, delta=delta)
    split_heuristic = top1_configure_split_heuristic(1)
    
end


∂bounds_radii = []
for r in [0.01, 0.03, 0.05]
    println("### r = ", r)
    ∂bounds_net  = verify_bounds_sample(X_test, nn, nets_l1_1e4[end], r, n_test)
    push!(∂bounds_radii, ∂bounds_net)
end

radii = [0.01, 0.03, 0.05]
p = plot(title="Verified bounds for different radii", xlabel="radius", ylabel="verified error")
for (i, bnds) in enumerate(∂bounds_radii)
    if i == 1
        scatter!(repeat([radii[i]], n_test), bnds, label="verified error (samples)", color=2)
    else
        scatter!(repeat([radii[i]], n_test), bnds, label="sample $i", primary=false, color=2)
    end
end
p



res = load("../VeryDiffPolyExperiments/results/mnist/mnist_verified_bounds_mnist_256x4_1e4_2025-04-08T21:48:01.008.jld2")
nn_polys = res["nets"];


function verify_eps_equiv(nn_poly, nn; z = nothing)
    z = isnothing(z) ? Zonotope(zeros(784), ones(784)) : z

    nn_diff = GeminiNetwork(nn_poly, nn);
    dom = VeryDiff.extract_approximation_domain(nn_poly)
    ∂z = Zonotope(zero(z.G), zero(z.c), nothing)
    zΔ = DiffZonotope(z, deepcopy(z), ∂z, 0, 0, 0)
    zΔ = nn_diff(zΔ, PropState(true), dom, nothing)
    return zΔ
end



x = X_test[9]
l = clamp.(x .- 0.05, 0, 1)
u = clamp.(x .+ 0.05, 0, 1)
z = Zonotope(l, u)

y_test[9]

zΔ_eps = verify_eps_equiv(nn_polys[15], nn, z=z)
zΔ = verify_eps_equiv(nn_polys[15], nn)
@show maximum(abs.(zono_bounds(zΔ_eps.∂Z)))
@show maximum(abs.(zono_bounds(zΔ.∂Z)))




cs = [0.010146330799075632, 0.016274850854871598, 0.0075410126468994695, 0.0005394232182953051, -0.0014623791991036704, -0.0003193753457596618, 0.0005944280012474311, 0.00022358544888859122, -0.00030560190985523875, -0.00016925575938067075, 0.00017484556065626485, 0.00013381546401223577, -0.00010498710372092295, -0.00010859174437651299, 6.363038582925456e-5, 8.954388327565187e-5, -3.743011330378146e-5, -7.453792021531672e-5, 2.005907501137283e-5, 6.234251228255028e-5, -8.19729302278317e-6, -5.2198538542450676e-5, -3.736372280620472e-8, 4.361322115156738e-5, 5.779633916209648e-6, -3.6253042080342086e-5, -9.749231309964107e-6, 2.9884236115006562e-5, 1.2422687907889132e-5, -2.433789998989325e-5, -1.412752280747303e-5, 1.948860801645845e-5, 1.5096396647798742e-5, -1.5240811074208284e-5, -1.5499544480721152e-5, 1.1519912039754869e-5, 1.546488143754251e-5, -8.266254111839335e-6, -1.5090840375561264e-5, 5.430981788048022e-6, 1.4454780167979965e-5, -2.973139955121745e-6, -1.3618615514518071e-5, 8.576123800098541e-7, 1.2632659098289012e-5, 9.463572631123651e-7, -1.1538289112742356e-5, -2.466237148586689e-6, 1.0369807083081509e-5, 3.7268958073418743e-6, -9.15610589694773e-6, -4.752491799058402e-6, 7.916175406839142e-6, 5.5445975412590825e-6, -6.750559426525953e-6, -6.452413916908309e-6, 4.333742948507608e-6, 2.024979013855549e-6, -2.2976893755951685e-5, -8.31788603757896e-5, 2.174817529562188e-5];
l̂ = -0.03836874833041293
û = 0.033246769646439134
l = -0.0015548316744029265
u = 0.00217382942388641

fc = VeryDiff.make_eval_chebyshev(cs, l̂, û)
cs2 = VeryDiff.chebyshev_coefficients(fc, l, u, length(cs)-1)
fc2 = VeryDiff.make_eval_chebyshev(cs2, l, u)


xs = range(l, u, 200)
plot(xs, fc.(xs), label="poly", framestyle=:origin)
plot!(xs, fc2.(xs), label="poly2")

rs = VeryDiff.chebyshev_roots(cs, l̂, û)
rs2 = VeryDiff.chebyshev_roots(cs2[1:end-1], l, u)

rs = rs[(rs .>= l) .& (rs .<= u)]
rs2 = rs2[(rs2 .>= l) .& (rs2 .<= u)]

scatter!(rs, fc.(rs), label="roots")
scatter!(rs2, fc.(rs2), label="roots2")




function cheb_coeffs(f, degree::Integer; kind=2)
    N = degree + 1
    if kind == 1
        # xk = [cos(π*(k + 0.5)/N) for k = 0:N-1]
        cs = [2/N * sum([f(cos(π*(k + 0.5) / N)) * cos(π*j*(k + 0.5) / N) for k =0:N-1]) for j = 0:N-1]

        # f(x) ≈ (2/n * ∑ₙ cₙ⋅Tₙ(x)) - 0.5*c₀
        # since T₀(x) = 1, we can just adjust that coefficient
        cs[1] -= 0.5*cs[1] 
    else
        # xk = [cos(k*π / degree) for k = 0:degree]
        # coded based on https://andrea-combette.com/post/spectral-chebyshex/
        # why are there so few resources on 2nd kind Chebyshev interpolation?
        c̄ = ones(degree + 1)
        c̄[1]   = 2
        c̄[end] = 2
        cs = [2/(degree*c̄[j+1]) * sum(f(cos(k*π/degree)) * cos(k*j*π/degree)/c̄[k+1] for k = 0:degree) for j = 0:degree]
    end
    cs
end








function print_diagnostics(z)
    println("Z1 -> G : ", size(z.Z₁.G), ", c : ", size(z.Z₁.c), ", influence : ", isnothing(z.Z₁.influence) ? "nothing" : size(z.Z₁.influence))
    println("Z2 -> G : ", size(z.Z₂.G), ", c : ", size(z.Z₂.c), ", influence : ", isnothing(z.Z₂.influence) ? "nothing" : size(z.Z₂.influence))
    println("∂Z -> G : ", size(z.∂Z.G), ", c : ", size(z.∂Z.c), ", influence : ", isnothing(z.∂Z.influence) ? "nothing" : size(z.∂Z.influence))
end


function fixed_relu_dist(l, u)
    if l > 0
        return 0.
    elseif u < 0
        return 0.
    else
        return min(abs(l), abs(u))
    end
end


