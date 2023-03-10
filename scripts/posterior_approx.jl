using KernelFunctions
using Plots
using Distributions
using LinearAlgebra
using Random

include("../src/itergp.jl")
isdir("plots") || mkdir("plots")
ylims = (-5, 5)

n = 500
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
cholesky_fit_plt = plot(cholesky_post, xx, title="Cholesky fit"; ylims)
scatter!(cholesky_fit_plt, x, y, label="Data", color=2)
savefig(cholesky_fit_plt, joinpath("plots", "cholesky_posterior.png"))

rank = 15
Σy = Diagonal(fill(σ², n))
x0 = zeros(length(Sx))
K = kernelmatrix(prior.kernel, Sx)
P = cholesky_preconditioner(K, rank, σ²)
A = K + σ²*I
b = Sy - prior.mean.(Sx)

cg_policy = ConjugateGradientPolicy(A, b, x0, n)
cg_post = posterior(cg_policy, prior, Sx, Sy);
cg_fit_plt = plot(cg_post, xx, title="Conjugate gradients fit"; ylims);
scatter!(cg_fit_plt, x, y, label="Data", color=2)
savefig(cg_fit_plt, joinpath("plots", "cg_posterior.png"))

# Yarr! Neede due to https://github.com/timholy/WoodburyMatrices.jl/issues/35
function WoodburyMatrices._ldiv!(dest, W::SymWoodbury, A::Union{Factorization, Diagonal}, B)
    WoodburyMatrices.myldiv!(W.tmpN1, A, B)
    mul!(W.tmpk1, W.V, W.tmpN1)
    mul!(W.tmpk2, W.Cp, W.tmpk1)
    mul!(W.tmpN2, W.U, W.tmpk2)
    WoodburyMatrices.myldiv!(A, W.tmpN2)
    for i = 1:length(W.tmpN2)
        @inbounds dest[i] = W.tmpN1[i] - W.tmpN2[i]
    end
    return dest
end

pcg_policy = PreconditionedConjugateGradientPolicy(A, b, x0, n, P)
pcg_post = posterior(pcg_policy, prior, Sx, Sy);
pcg_fit_plt = plot(pcg_post, xx, label=nothing, title="Preconditioned conjugate gradients fit"; ylims)
scatter!(pcg_fit_plt, x, y, label="Data", color=2)
savefig(pcg_fit_plt, joinpath("plots", "precond_cg_posterior.png"))
isapprox(pcg_post.C, cg_post.C, rtol=1e-3, atol=1e-3)