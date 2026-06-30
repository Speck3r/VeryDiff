

""" 
    `dgelu_bounds(l, u)`

    Compute the bounds on the derivative of GeLU(x) for x in [l, u].

    The derivative of GeLU is piecewise defined with three pieces:
    - For x <= -sqrt(2), dGeLU(x) is monotonically decreasing
    - For -sqrt(2) < x < sqrt(2), dGeLU(x) is monotonically increasing
    - For x >= sqrt(2), dGeLU(x) is monotonically decreasing

    Therefore, to compute the bounds on the derivative, we need to consider the intersection of the interval [l, u] with these three pieces.

    args:
    - `l`: lower bound of the interval
    - `u`: upper bound of the interval

    returns:
    - `∂l`: lower bound on the derivative of GeLU in [l, u]
    - `∂u`: upper bound on the derivative of GeLU in [l, u]
"""
function dgelu_bounds(l, u)
    ∂l₁, ∂l₂, ∂l₃ =  Inf,  Inf,  Inf
    ∂u₁, ∂u₂, ∂u₃ = -Inf, -Inf, -Inf

    if l <= min(u, -sqrt(2))
        # intersect with first interval
        ∂l₁ = dgelu(-sqrt(2))
        ∂u₁ = dgelu(l)
    end
    if max(l, -sqrt(2)) <= min(u, sqrt(2))
        # intersect with second interval
        ∂l₂ = dgelu(max(l, -sqrt(2)))
        ∂u₂ = dgelu(min(u, sqrt(2)))
    end
    if max(l, sqrt(2)) <= u
        # intersect with third interval
        ∂l₃ = dgelu(u)
        ∂u₃ = dgelu(max(l, sqrt(2)))
    end

    ∂l = min(∂l₁, ∂l₂, ∂l₃)
    ∂u = max(∂u₁, ∂u₂, ∂u₃)

    return ∂l, ∂u
end


"""
    `gelu_diff_relax_parallel(lx, ux, ly, uy, lΔ, uΔ)`

    Compute parallel linear relaxation for GeLU(x) - GeLU(y) = GeLU(x) - GeLU(x - Δ).
    The relaxation is of the form

    a*Δ + b - ϵ ≤ GeLU(x) - GeLU(x - Δ) ≤ a*Δ + b + ϵ

    where a, b, ϵ are computed by this function.

    The relaxation is constructed as a parallel version of the relaxation proposed in "Input-Relataional Verification of Neural Networks".

    args:
    - `lx`, `ux`: lower and upper bound for x
    - `ly`, `uy`: lower and upper bound for y
    - `lΔ`, `uΔ`: lower and upper bound for Δ

    returns:
    - `a`, `b`, `ϵ`: parameters of the parallel linear relaxation
"""
function gelu_diff_relax_parallel(lx, ux, ly, uy, lΔ, uΔ)
    l, u = min(lx, ly), max(ux, uy)
    ∂l, ∂u = dgelu_bounds(l, u)

    x₀ = 0
    f1 = x -> ∂u*(x - x₀) + gelu(x₀)
    f2 = x -> ∂l*(x - x₀) + gelu(x₀)

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