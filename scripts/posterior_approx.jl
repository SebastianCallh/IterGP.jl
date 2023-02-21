using KernelFunctions
using Plots
using Distributions
using LinearAlgebra
using Random

include("../src/itergp.jl")
isdir("plots") || mkdir("plots")

n = 100
σ² = 0.25
rng = MersenneTwister(1234)
x, y = sinusoid(rng, n, sqrt(σ²))
xx = collect(1.4 .* range(extrema(x)..., 200))

k = Matern32Kernel()
prior = GP(Returns(0.), k, σ²)
prior_plt = plot(prior, xx)
scatter!(prior_plt, x, y, color=2, label="Data")

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
Sx = S*x
Sy = S*y

cholesky_policy = CholeskyPolicy(length(x), min(n, 300))
cholesky_post = posterior(cholesky_policy, prior, Sx, Sy)
cholesky_fit_plt = plot(cholesky_post, xx, title="Cholesky fit")
scatter!(cholesky_fit_plt, x, y, label="Data", color=2)
savefig(cholesky_fit_plt, joinpath("plots", "cholesky_posterior.png"))

rank = 100
x0 = rand(length(x))
K = kernelmatrix(prior.kernel, Sx)
A = K + σ²*I
b = Sy - prior.mean.(Sx)
P = cholesky_preconditioner(K, rank, σ²)
cond(A)
cond(P\A)

cg_policy = ConjugateGradientPolicy(A, b, x0, P, n, 1e-6, 1e-6);
cg_post = posterior(cg_policy, prior, Sx, Sy);
cg_fit_plt = plot(cg_post, xx, title="Conjugate gradients fit")
scatter!(cg_fit_plt, x, y, label="Data", color=2)
savefig(cg_fit_plt, joinpath("plots", "cg_posterior.png"))