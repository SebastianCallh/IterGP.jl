using Revise
using KernelFunctions
using Plots
using Distributions
using LinearAlgebra
using Random

include("src/itergp.jl")
using .IterGP

rng = MersenneTwister(1234)
x = Float64.(collect(range(-4, 4, 50)))
xx = collect(1.4 .* range(extrema(x)..., 200))
y = sin.(x) .+ 0.25*randn(rng, length(x))
k = Matern32Kernel()
σ² = 0.25

prior = GP(Returns(0.), k, σ²)
plot(prior, xx)

policy = CholeskyPolicy(length(x), 3)
cholesky_post = posterior(policy, prior, x, y)
cholesky_fit = plot(cholesky_post, xx)
scatter!(cholesky_fit, x, y, label="Data", color=2)

A = kernelmatrix(prior.kernel, x, x) + prior.σ²*I
b = y - prior.mean.(x)
policy = ConjugateGradientPolicy(A, b, rand(length(x)), 1, 1e-6, 1e-6)
cg_post = posterior(policy, prior, x, y)
gc_fit = plot(cg_post, xx)
scatter!(gc_fit, x, y, label="Data", color=2)