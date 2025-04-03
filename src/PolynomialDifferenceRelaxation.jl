

"""
Test if the point (x,y) is contained in the box [lx, ux] × [ly, uy].
"""
function in_box(x, y, lx, ux, ly, uy)
    (lx <= x) && (x <= ux) && (ly <= y) && (y <= uy)
end


"""
Computes the boundary points of H ∩ {x ≤ y} and H ∩ {x ≥ y} sorted in counter-clockwise order.

Tests for each corner of H and for each possible intersection with x = y in counter-clockwise order
in which part of the half-space the point is included and adds it to the respective list.

Note that one of the vectors can be empty, e.g. if H is [3,4] × [0, 1]

args:
    lx - lower x value of H
    ux - upper x value of H
    ly - lower y value of H
    uy - upper y value of H

returns:
    (ps_leq, ps_geq) - vectors of points [p1, p2, ..., pn, p1] (vectors can be empty!)
"""
function get_boundary_sorted(lx::N, ux::N, ly::N, uy::N) where N<:Number
    # H is hyperrectangle [lx,ux] × [ly,uy]
    ps_leq = Vector{Tuple{N,N}}()  # boundary for H ∩ {x ≤ y}
    ps_geq = Vector{Tuple{N,N}}()  # boundary for H ∩ {x ≥ y}

    function test_point(p)
        # test on which side of x = y the points is
        # only add to list, if it is not a duplicate
        (x,y) = p
        if x <= y 
            if length(ps_leq) > 0
                (x̂, ŷ) = ps_leq[end]
                ((x̂ != x) || (ŷ != y)) && push!(ps_leq, (x, y))
            else
                push!(ps_leq, (x, y))
            end
        end
        if x >= y
            if length(ps_geq) > 0
                (x̂, ŷ) = ps_geq[end]
                ((x̂ != x) || (ŷ != y)) && push!(ps_geq, (x, y))
            else
                push!(ps_geq, (x, y))
            end
        end
    end
    
    test_point((lx, ly))
    # intersection with y = ly
    if in_box(ly, ly, lx, ux, ly, uy)
        test_point((ly, ly))
    end
    test_point((ux, ly))
    # intersection with x = ux
    if in_box(ux, ux, lx, ux, ly, uy)
        test_point((ux, ux))
    end
    test_point((ux, uy))
    # intersection with y = uy
    if in_box(uy, uy, lx, ux, ly, uy)
        test_point((uy, uy))
    end
    test_point((lx, uy))
    # intersection with x = lx
    if in_box(lx, lx, lx, ux, ly, uy)
        test_point((lx, lx))
    end

    
    #if length(ps_leq) == 0
    #    println("lx = ", lx, ", ux = ", ux, ", ly = ", ly, ", uy = ", uy)
    #    println("ps_leq = ", ps_leq)
    #    println("ps_geq = ", ps_geq)
    #end

    # want points to be [p1, p2, ..., pn, p1]
    if !isempty(ps_leq) && (ps_leq[end] != ps_leq[1])
        push!(ps_leq, ps_leq[1])
    end

    if !isempty(ps_geq) && (ps_geq[end] != ps_geq[1])
        push!(ps_geq, ps_geq[1])
    end

    return ps_leq, ps_geq
end


"""
Computes critical points of p(x) - ReLU(x - Δ) - (ax + bΔ) in the inactive ReLU case, i.e.
the critical points of p(x) - (ax + bΔ).

args:
    boundary - vector [v₀, v₁, ..., vₙ, v₀] of counter-clockwise sorted 2d points representing the 
               polygonal boundary of the input set for (x, Δ).
    ps - vector of monomial coefficients for p(x) in order [p₀, p₁, ...]
    a - coefficient of x in the linear relaxation
    b - coefficient of Δ in the linear relaxation 

returns:
    vector of 2d critical points 
"""
function relax_diff_inact(boundary, ps, a::N, b::N) where N<:Number
    dps = dpoly(ps)
    # d/dx p(x) - ax
    dpa = copy(dps)
    dpa[1] -= a
    # all boundary points can be critical points
    critical_points = copy(boundary)
    if b == 0
        # extremum in interior is only possible if b == 0
        lx = minimum(first.(boundary))
        ux = maximum(first.(boundary))
        lΔ = minimum(last.(boundary))
        uΔ = maximum(last.(boundary))
        # if the derivative is the 0 polynomial, then the original polynomial is constant
        # so it doesn't matter where we evaluate it, just choose some point within the bounds
        rs = all(dpa .== 0) ? [0.5 * (lx + ux)] : real_roots(dpa)
        rs = rs[(lx .<= rs) .& (rs .<= ux)]
        # if b == 0, we have p(x) - ax and we get the same value everywhere regardless of Δ, 
        # so we just need to choose a feasible value of Δ.
        # since we are in the x ≤ Δ case, we need Δ to be at least as large as x
        length(rs) > 0 && push!(critical_points, [(r, clamp(0.5*(lΔ + uΔ), r, uΔ)) for r in rs]...)
    end

    # boundary
    for ((x0, Δ0), (x1, Δ1)) in zip(boundary[1:end-1], boundary[2:end])
        lxi = min(x0, x1)
        uxi = max(x0, x1)
        lΔi = min(Δ0, Δ1)
        uΔi = max(Δ0, Δ1)
        
        if x0 == x1
            # x is constant 
            # f(x, Δ) = p(x) - ax - bΔ is monotonic in Δ
            push!(critical_points, (x0, Δ0))
            push!(critical_points, (x0, Δ1))
        elseif Δ0 == Δ1
            # Δ is constant
            # f(x, Δ) = p(x) - ax - bΔ is just a function of x
            xs = all(dpa .== 0) ? [0.5 * (lxi + uxi)] : real_roots(dpa)          
            xs = xs[(lxi .<= xs) .& (xs .<= uxi)]
            if length(xs) > 0
                for x in xs
                    push!(critical_points, (x, Δ0))
                end
            end
        else
            # we are on the x == Δ part of the boundary
            # f(x, Δ) = f(t) = p(t) - (a + b)t
            dpab = copy(dps)
            dpab[1] -= a + b
            ts = all(dpab .== 0) ? [0.5 * (max(lxi, lΔi) + min(uxi, uΔi))] : real_roots(dpab)
            ts = ts[(lxi .<= ts) .& (ts .<= uxi) .& (lΔi .<= ts) .& (ts .<= uΔi)]
            if length(ts) > 0
                for t in ts
                    push!(critical_points, (t, t))
                end
            end
        end
    end

    return critical_points
end 


# for relax_diff_inact, we have f(x, Δ) = p(x) - ax - bΔ
# for relax_diff_act, we have f(x, Δ) = p(x) - x + Δ - ax - bΔ = p(x) - (1+a)x - (1-b)Δ
# so we can just reuse the inact case
relax_diff_act(boundary, ps, a, b) = relax_diff_inact(boundary, ps, 1+a, 1-b)


"""
Computes bias and coefficient of the new error term for a parallel
linear relaxation of f(x, Δ) = p(x) - ReLU(x - Δ).

The final relaxation guarantees
    a⋅x + b⋅Δ + c - ϵ ≤ f(x, Δ) ≤ a⋅x + b⋅Δ + c + ϵ

args:
    lx - concrete lower bound on x
    ux - concrete upper bound on x
    lΔ - concrete lower bound on Δ
    uΔ - concrete upper bound on Δ
    ps - coefficients of polynomial p(x) = p₀ + p₁x + ...
         in order [p₀, p₁, ...]
    a  - slope for x in linear relaxation
    b  - slope for Δ in linear relaxation

returns:
    c  - bias for linear relaxation
    ϵ  - coefficient for new error term in linear relaxation
"""
function parallel_diff_relaxation(lx, ux, lΔ, uΔ, ps, a, b)
    #eval_poly = x -> sum(ps[k]*x^(k-1) for k in 1:length(ps))
    eval_poly = make_eval_poly(ps)
    f = (x,Δ) -> eval_poly(x) - max.(0, x - Δ) - a*x - b*Δ
    
    ps_leq, ps_geq = get_boundary_sorted(lx, ux, lΔ, uΔ)

    # ReLU(x - Δ) inactive, if x ≤ Δ
    if !isempty(ps_leq)
        crit_leq = relax_diff_inact(ps_leq, ps, a, b)
        ϵs_leq = f.(first.(crit_leq), last.(crit_leq))
    else
        ϵs_leq = []
    end

    if !isempty(ps_geq)
        # ReLU(x - Δ) active, if x ≥ Δ
        crit_geq = relax_diff_act(ps_geq, ps, a, b)
        ϵs_geq = f.(first.(crit_geq), last.(crit_geq))
    else
        ϵs_geq = []
    end
    #crit_leq = relax_diff_inact(ps_leq, ps, a, b)
    #crit_geq = relax_diff_act(ps_geq, ps, a, b)

    #ϵs_leq = f.(first.(crit_leq), last.(crit_leq))
    #ϵs_geq = f.(first.(crit_geq), last.(crit_geq))
   
    ϵ_max = max(maximum(ϵs_leq, init=-Inf), maximum(ϵs_geq, init=-Inf))
    ϵ_min = min(minimum(ϵs_leq, init=Inf), minimum(ϵs_geq, init=Inf))

    c = 0.5*(ϵ_min + ϵ_max)
    ϵ = 0.5*(ϵ_max - ϵ_min)

    return c, ϵ
end


"""
Use an optimization algorithm to find a parallel relaxation of p(x) - ReLU(x - Δ) with small absolute error.

The relaxation has form ax + bΔ + c - ϵ ≤ p(x) - ReLU(x - Δ) ≤ ax + bΔ + c + ϵ.

args:
    lx - concrete lower bound on x
    ux - concrete upper bound on x
    lΔ - concrete lower bound on Δ
    uΔ - concrete upper bound on Δ
    ps - vector of monomial coefficients of p in order [p₀, p₁, ...]

optional args:
    a - initial value for coefficient of x in the linear relaxation 
    b - initial value for coefficient of Δ in the linear relaxation 

kwargs:
    alg - algorithm used for optimization (default: NelderMead) (any algorithm from Optim.jl can be used)
    options - Optim.Options() for passing parameters to the optimizer

returns:
    a - coefficient of x in the linear relaxation 
    b - coefficient of Δ in the linear relaxation
    c - bias of the linear relaxation 
    ϵ - coefficient of the new error term in the linear relaxation
"""
function find_good_poly_diff_approx(lx, ux, lΔ, uΔ, ps, a=0., b=0.; alg=NelderMead(), options=Optim.Options())
    a_best = [a, b]
    if options.iterations > 0
        optfun = a -> parallel_diff_relaxation(lx, ux, lΔ, uΔ, ps, a[1], a[2])[2]

        res = optimize(optfun, a_best, alg, options)
        a_best = Optim.minimizer(res)
    end

    c, ϵ = parallel_diff_relaxation(lx, ux, lΔ, uΔ, ps, a_best[1], a_best[2])

    return a_best[1], a_best[2], c, ϵ
end