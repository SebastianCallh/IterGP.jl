using KernelFunctions
using Plots
using Distributions
using LinearAlgebra
using Random

include("../src/itergp.jl")
using .IterGP
isdir("plots") || mkdir("plots")

rng = MersenneTwister(1234)
n = 500
x = Float64.(collect(range(-4, 4, n)))
xx = collect(1.4 .* range(extrema(x)..., 200))
y = sin.(x) .+ 0.25*randn(rng, length(x))
k = Matern32Kernel()
σ² = 0.25
prior = GP(Returns(0.), k, σ²)
plot(prior, xx)

cholesky_policy = CholeskyPolicy(length(x), 50)
cholesky_post = posterior(cholesky_policy, prior, x, y)
cholesky_fit_plt = plot(cholesky_post, xx, title="Cholesky fit")
scatter!(cholesky_fit_plt, x, y, label="Data", color=2)
savefig(cholesky_fit_plt, joinpath("plots", "cholesky_posterior.png"))

function perm(n)
    m = n
    A = zeros(Int,m,n)
    rowind = shuffle(1:m)
    for (j,i) in enumerate(rowind)
        A[i,j] = 1
    end
    A 
end
S = perm(n)
xperm = S*x
yperm = S*y
Prank = 100
x0 = rand(length(x))
K = kernelmatrix(prior.kernel, xpperm)
A = K + σ²*I
b = yperm - prior.mean.(xperm)
P = cholesky_preconditioner(K, Prank, σ²)
cond(A)
cond(P\A)

cg_policy = ConjugateGradientPolicy(A, b, x0, P, n, 1e-6, 1e-6);
cg_post = posterior(cg_policy, prior, xperm, yperm);
cg_fit_plt = plot(cg_post, xx, title="Conjugate gradients fit")
scatter!(cg_fit_plt, x, y, label="Data", color=2)
savefig(cg_fit_plt, joinpath("plots", "cg_posterior.png"))