
# Try to re-implement the algorithm described in 
#      A Provably Componentwise Backward Stable O(n²) QR Algorithm for the Diagnoalization of Colleague Matrices 
#      Serkh, Rokhlin 2021
#      https://arxiv.org/pdf/2102.12186



"""
Compute a complex rotation matrix for a 2D vector eliminating its first entry.

If the vector x = (x₁, x₂) is not zero, the rotation matrix R is such that R * x = (0, y) for some y.

args:
    x - 2D vector
returns:
    R - 2D rotation matrix
"""
function rotation_matrix(x::AbstractVector)
    @assert length(x) == 2 "Only 2D rotation matrix is supported"
    r = norm(x, 2)
    if r == 0
        c = 1.
        s = 0.
    else
        c = x[2] / r
        s = x[1] / r
    end

    return [c -s; conj(s) conj(c)]
end


"""
Given a (2 × 2) rotation matrix Qk, construct a (2 × 2) rotation matrix for the (k, k-1) entries of an n-dimensional vector.
"""
function rotation_matrix(Qk::AbstractMatrix, k::Integer, n::Integer)
    Uk = Matrix(Float64.(I(n)))
    Uk[k-1, k-1] = Qk[1, 1]
    Uk[k-1, k] = Qk[1, 2]
    Uk[k, k-1] = Qk[2, 1]
    Uk[k, k] = Qk[2, 2]

    return Uk
end


function rotation_matrices(Qs::AbstractMatrix)
    n = Int(size(Qs, 2) / 2) + 1
    Us = [rotation_matrix(Qs[:, 2*(k-1)-1:2*(k-1)], k, n) for k in 2:n]
end


function construct_hermite(d::AbstractVector, β::AbstractVector)
    n = length(d)
    @assert length(β) == n - 1 "β (the superdiagonal) must be of length n - 1"

    A = zeros(n, n)
    for i in 1:n
        A[i, i] = d[i]
        if i < n
            A[i, i+1] = β[i]
            A[i+1, i] = conj(β[i])
        end
    end

    return A
end


function construct_complete_matrix(d::AbstractVector, β::AbstractVector, p::AbstractVector, q::AbstractVector)
    n = length(d)
    @assert length(β) == n - 1 "β (the superdiagonal) must be of length n - 1"
    @assert length(p) == n "p must be of length n"
    @assert length(q) == n "q must be of length n"

    A = construct_hermite(d, β)
    B = A .+ p * conj.(q)'

    return B    
end


"""
Eliminate the superdiagonal of the lower Hessenberg matrix A + pq* where d is the diagonal and β the superdiagonal of A.

The algoritm returns rotation matrices Q₂, Q₃, ..., Qₙ such that, when Uₖ be the matrix that rotates the (k-1, k) plane by Qₖ, then U₂U₃ ⋯ Uₙ(A + pq*) is lower triangular.
Also returns vectors d, γ where 
    d is the diagonal of the resulting matrix A and 
    γ is the subdiagonal of the resulting matrix.

args:
    d - diagonal of the matrix A
    β - superdiagonal of the matrix A
    p - vector
    q - vector 

returns:
    Qs - (2 × 2(n-1)) matrix s.t. Qₖ = Qs[:, 2*(k-1)-1:2*(k-1)] rotation matrices
    d - diagonal of the resulting matrix U₂U₃ ⋯ UₙA
    γ - subdiagonal of the resulting matrix U₂U₃ ⋯ UₙA
    p - modified original vector p using U₂U₃ ⋯ Uₙp
"""
function eliminate_superdiagonal(d::AbstractVector, β::AbstractVector, p::AbstractVector, q::AbstractVector)
    n = length(d)
    @assert length(β) == n - 1 "β (the superdiagonal) must be of length n - 1"
    @assert length(p) == n "p must be of length n"
    @assert length(q) == n "q must be of length n"


    γ = conj.(β)
    q̃ = copy(q)

    Qs = zeros(2, 2*(n-1))
    for k in n:-1:2
        Qk = rotation_matrix([β[k-1] + p[k-1]*conj(q[k]) , d[k] + p[k]*conj(q[k])])

        if k != 2
            # rotate subdiagonal and the sub-subdiagonal
            γ[k-2] = (Qk * [γ[k-2], -q̃[k]*conj(p[k-2])])[1]
        end

        # rotate the diagonal and the subdiagonal
        y = Qk * [d[k-1], γ[k-1]]
        d[k-1] = y[1]
        γ[k-1] = y[2]

        # rotate the superdiagonal and the diagonal
        y = Qk * [β[k-1], d[k]]
        β[k-1] = y[1]
        d[k] = y[2]

        # rotate p 
        y = Qk * [p[k-1], p[k]]
        p[k-1] = y[1]
        p[k] = y[2]

        if abs(p[k-1]*conj(q[k]))^2 + abs(p[k]*conj(q[k]))^2 > abs(β[k-1])^2 + abs(d[k])^2
            # correct p
            p[k-1] = - β[k-1] / conj(q[k])
        end

        # rotate q̃
        y = Qk * [q̃[k-1], q̃[k]]
        q̃[k-1] = y[1]
        q̃[k] = y[2]

        Qs[:, 2*(k-1)-1:2*(k-1)] = Qk
    end

    return Qs, d, γ, p 
end


"""
Rotates the lower triangular matrix B + pq* back into Hessenberg form.

The matrix B is given by its diagonal d and subdiagonal γ.
And the rotation matrices Qₖ are such that Uₖ rotates the (k, k-1) plane by Qₖ.

The algorithm returns the vectors d, β, q describing the matrix BUₙ*⋯U₂*, where
    d is the diagonal of the resulting matrix B and 
    β is the superdiagonal of the resulting matrix.
    q is the modification of the original vector q as q = U₂*⋯Uₙ*q.
"""
function rotate_to_hessenberg(Qs::AbstractMatrix, d::AbstractVector, γ::AbstractVector, p::AbstractVector, q::AbstractVector)
    n = length(d)
    @assert size(Qs, 2) == 2*(n-1) "Qs must be of size (2, 2(n-1))"
    @assert size(Qs, 1) == 2 "Qs must be of size (2, 2(n-1))"
    @assert length(γ) == n - 1 "γ must be of length n - 1"
    @assert length(p) == n "p must be of length n"
    @assert length(q) == n "q must be of length n"

    β = zeros(n-1)
    for k in n:-1:2
        # rotate the diagonal and the superdiagonal
        Qk = Qs[:, 2*(k-1)-1:2*(k-1)]
        y = conj.(Qk) * [d[k-1], -p[k-1]*conj(q[k])]
        d[k-1] = y[1]
        β[k-1] = y[2]

        # rotate the subdiagonal and the diagonal
        y = conj.(Qk) * [γ[k-1], d[k]]
        d[k] = y[2]

        # rotate q 
        y = Qk * [q[k-1], q[k]]
        q[k-1] = y[1]
        q[k] = y[2]
    end

    return d, β, q
end


function qr_unshifted(d::AbstractVector, β::AbstractVector, p::AbstractVector, q::AbstractVector; tol=1e-10, max_iter=1000)
    n = length(d)
    @assert length(β) == n - 1 "β (the superdiagonal) must be of length n - 1"
    @assert length(p) == n "p must be of length n"
    @assert length(q) == n "q must be be of length n"


    for i in 1:n-1
        for j in 1:max_iter
            if β[i] + p[i]*conj(q[i+1]) <= tol
                # eigenvalue tolerance reached
                break
            end


            Qs, d̂, γ, p̂ = eliminate_superdiagonal(d[i:n], β[i:n-1], p[i:n], q[i:n])
            d̂, β̂, q̂ = rotate_to_hessenberg(Qs, d̂, γ, p̂, q[i:n])
            # TODO: is this assignment correct?
            d[i:n] .= d̂
            β[i:n-1] .= β̂
            q[i:n] .= q̂
        end
    end

    return d
end


function monic_colleague_matrix(cs)
    cs = 1/cs[end] .* cs
    cs = cs[1:end-1]
    n = length(cs)
    @assert n > 1 "Colleague matrix must be of size > 1"

    # diagonal
    d = zeros(n)
    
    # superdiagonal
    β = 0.5 .* ones(n-1)
    β[1] = 1 / sqrt(2)

    # p and q are the same
    p = zeros(n)
    p[end] = 1.

    q = -0.5 .* cs
    q[1] *= sqrt(2)

    return d, β, p, q 
end





# testing

function generate_random_decomposable_matrix(n)
    cs = VeryDiff.chebyshev_coefficients(x -> max(0, x), -1., 1., n)
    monic_colleague_matrix(cs)
end


"""
The function eliminate_superdiagonal promises that U₂U₃ ⋯ Uₙ(A + pq*) is lower triangular.
"""
function test_lower_triangular(d, β, p, q; tol=1e-14)
    n = length(d)

    # generate A + pq*
    H = construct_complete_matrix(d, β, p, q)

    Qs, d, γ, p = eliminate_superdiagonal(d, β, p, q)

    for i in n:-1:2
        Qk = Qs[:, 2*(i-1)-1:2*(i-1)]
        Uk = rotation_matrix(Qk, i, n)
        H = Uk * H
    end

    # check if H is lower triangular
    for i in 1:n
        for j in i+1:n
            if abs(H[i, j]) > tol
                println("H is not lower triangular! H[$i, $j] = ", H[i, j])
                return false
            end
        end
    end
    return true
end


"""
The function eliminate_superdiagonal gurantees that d is the diagonal and γ the subdiagonal of U₂U₃ ⋯ UₙA
"""
function test_diag_and_subdiag(d, β, p, q; tol=1e-14)
    n = length(d)

    # generate A
    A = construct_hermite(d, β)

    Qs, d, γ, p = eliminate_superdiagonal(d, β, p, q)

    for i in n:-1:2
        Qk = Qs[:, 2*(i-1)-1:2*(i-1)]
        Uk = rotation_matrix(Qk, i, n)
        A = Uk * A
    end

    # check diagonal
    for i in 1:n
        if abs(A[i, i] - d[i]) > tol
            println("Diagonal mismatch! A[$i, $i] = ", A[i, i], " != ", d[i])
            return false
        end
    end

    # check subdiagonal
    for i in 1:n-1
        if abs(A[i+1, i] - γ[i]) > tol
            println("Subdiagonal mismatch! A[$i+1, $i] = ", A[i+1, i], " != ", γ[i])
            return false
        end
    end

    return true
end





