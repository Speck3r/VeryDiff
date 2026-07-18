""" 
    `σ´_bounds(l, u)`

    Compute the bounds on the derivative of Sigmoid(x) for x in [l, u].

    The derivative of Sigmoid is piecewise defined with two pieces:
    - For x <= 0, σ´(x) is monotonically increasing
    - For x > 0, σ´(x) is monotonically decreasing

    Therefore, to compute the bounds on the derivative, we need to consider the intersection of the interval [l, u] with these two pieces.

    args:
    - `l`: lower bound of the interval
    - `u`: upper bound of the interval

    returns:
    - `∂l`: lower bound on the derivative of Sigmoid in [l, u]
    - `∂u`: upper bound on the derivative of Sigmoid in [l, u]
"""
function σ´_bounds(l, u)

    if l <= 0 && 0 <= u
        ∂l = min(σ´(l),σ´(u))
        ∂u = σ´(0)
    elseif u <= 0
        ∂l = σ´(l)
        ∂u = σ´(u)
    elseif 0 <= l
        ∂l = σ´(u)
        ∂u = σ´(l)
    end
    return ∂l, ∂u
end

"""
    `sigmoid_diff_relax_parallel(lx, ux, ly, uy, lΔ, uΔ)`

    Compute parallel linear relaxation for Sigmoid(x) - Sigmoid(y) = Sigmoid(x) - Sigmoid(x - Δ).
    The relaxation is of the form

    a*Δ + b - ϵ ≤ Sigmoid(x) - Sigmoid(x - Δ) ≤ a*Δ + b + ϵ

    where a, b, ϵ are computed by this function.

    The relaxation is constructed as a parallel version of the relaxation proposed in "Input-Relataional Verification of Neural Networks".

    args:
    - `lx`, `ux`: lower and upper bound for x
    - `ly`, `uy`: lower and upper bound for y
    - `lΔ`, `uΔ`: lower and upper bound for Δ

    returns:
    - `a`, `b`, `ϵ`: parameters of the parallel linear relaxation
"""
function sigmoid_diff_relax_parallel(lx, ux, ly, uy, lΔ, uΔ)
    l, u = min(lx, ly), max(ux, uy)
    ∂l, ∂u = σ´_bounds(l, u)

    #x₀ = 0
    f1 = x -> ∂u*(x) #∂u*(x - x₀) + σ(x₀)
    f2 = x -> ∂l*(x) #∂l*(x - x₀) + σ(x₀)

    if lΔ < 0 && uΔ > 0
        aₗ = (f2(uΔ) - f1(lΔ)) / (uΔ - lΔ)
        bₗ = -lΔ * aₗ + f1(lΔ)
    
        aᵤ = (f1(uΔ) - f2(lΔ)) / (uΔ - lΔ)
        bᵤ = -lΔ * aᵤ + f2(lΔ)

        a = 0.5 * (aₗ + aᵤ)
        # shift a*Δ up to be larger than aᵤ*Δ + bᵤ
        # so need
        # max (aᵤ - a)*Δ + bᵤ  over Δ in [lΔ, uΔ]
        # which is just (aᵤ - a)*lΔ + bᵤ if (aᵤ - a) is negative, and 
        # (aᵤ - a)*uΔ + bᵤ if (aᵤ - a) is positive
        bᵤ = ifelse(aᵤ - a >= 0, (aᵤ - a)*uΔ + bᵤ, (aᵤ - a)*lΔ + bᵤ)
        # same for lower bound
        bₗ = ifelse(aₗ - a >= 0, (aₗ - a)*lΔ + bₗ, (aₗ - a)*uΔ + bₗ)
    elseif lΔ >= 0
        # f1 is upper bound, f2 is lower bound
        aₗ = ∂l
        bₗ = -lΔ * aₗ + f2(lΔ)

        aᵤ = ∂u
        bᵤ = -lΔ * aᵤ + f1(lΔ)

        a = 0.5 * (aₗ + aᵤ)
        bᵤ = ifelse(aᵤ - a >= 0, (aᵤ - a)*uΔ + bᵤ, (aᵤ - a)*lΔ + bᵤ)
        bₗ = ifelse(aₗ - a >= 0, (aₗ - a)*lΔ + bₗ, (aₗ - a)*uΔ + bₗ)
    else uΔ <= 0
        # f1 is lower bound, f2 is upper bound
        aₗ = ∂u
        bₗ = -lΔ * aₗ + f1(lΔ)

        aᵤ = ∂l
        bᵤ = -lΔ * aᵤ + f2(lΔ)

        a = 0.5 * (aₗ + aᵤ)
        bᵤ = ifelse(aᵤ - a >= 0, (aᵤ - a)*uΔ + bᵤ, (aᵤ - a)*lΔ + bᵤ)
        bₗ = ifelse(aₗ - a >= 0, (aₗ - a)*lΔ + bₗ, (aₗ - a)*uΔ + bₗ)
    end

    b = 0.5*(bₗ + bᵤ)
    ϵ = 0.5*(bᵤ - bₗ)

    return a, b, ϵ
end