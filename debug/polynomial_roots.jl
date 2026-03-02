
using LinearAlgebra, VeryDiff, Plots



p, ϵ = VeryDiff.approx_gelu_poly(-5., 5., 100, verbosity=1)

C = VeryDiff.colleague_matrix(p)

symlog = x -> sign(x) * log(1 + abs(x))

heatmap(symlog.(C), yflip=true, title="Symlog of Colleague Matrix", xlabel="Column Index", ylabel="Row Index", colormap=:balance)

iters = 100
anim = @animate for i in 1:iters
    global C
    Q, R = qr(C)
    C = R * Q
    # heatmap(symlog.(C), yflip=true, title="Symlog of Colleague Matrix - Iteration $i", xlabel="Column Index", ylabel="Row Index", clims=(-5, 5), colormap=:balance)
    heatmap(log10.(abs.(C) .+ 1e-14), yflip=true, title="QR Iteration of Colleague Matrix - Iteration $i", xlabel="Column Index", ylabel="Row Index", colormap=:balance)
end

gif(anim, "colleague_matrix_iterations.gif", fps=1)


## Shifted QR algorithm for eigenvalues

C = VeryDiff.colleague_matrix(p)

iters = 100
anim_shifted = @animate for i in 1:iters
    global C
    # Shift using the bottom-right element
    shift = C[end, end]
    Q, R = qr(C - shift * I)
    C = R * Q + shift * I
    heatmap(log10.(abs.(C) .+ 1e-14), yflip=true, title="Shifted QR of Colleague Matrix - Iteration $i", xlabel="Column Index", ylabel="Row Index", colormap=:balance)
end

gif(anim_shifted, "colleague_matrix_shifted_iterations.gif", fps=1)