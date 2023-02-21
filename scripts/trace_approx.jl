using LinearAlgebra
using Distributions
using KernelFunctions
using Plots
using Random

include("../src/itergp.jl")
using .IterGP
isdir("plots") || mkdir("plots")

rng = MersenneTwister(1234)
n = 1000
x = Float64.(collect(range(-4, 4, n)))
xx = collect(1.4 .* range(extrema(x)..., 200))
y = sin.(x) .+ 0.25*randn(rng, length(x))
k = Matern32Kernel()
σ² = 0.25
ℓ = 100
K = kernelmatrix(k, x)
A = K + σ²*I
P = Matrix(cholesky_preconditioner(K, ℓ, σ²))
ℓs = 1:ℓ
pz̃ = MvNormal(zeros(n), I)
zs = rand(pz̃, ℓ) ./ sqrt(n)

approxs = map(i -> trace_approx(log(A), view(zs, :,1:i)), ℓs)
precond_approxs = map(i -> log_det_approx(A, P, view(zs, :,1:i)), ℓs)
logdet_plt = hline(
    [logdet(A)],
    color=:grey,
    label="True log(det(A))",
    style=:dash,
    linewidth=3,
    xlabel="No. random vectors ℓ",
    ylabel="Log det",
    title="Log determinant approximation"
)
plot!(logdet_plt, ℓs, approxs, color=1,linewidth=2, label="Approximation",)
plot!(logdet_plt, ℓs, precond_approxs, color=2, linewidth=2, label="Approximation precond.", )
savefig(logdet_plt, joinpath("plots", "logdet_approx.png"))