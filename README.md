# IterGPs

A (WIP) Julia implementation of [Posterior and Computational Uncertainty
in Gaussian Processes](https://arxiv.org/pdf/2205.15449.pdf). If this is interesting to you open a PR!

[![Build Status](https://github.com/SebastianCallh/IterGP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SebastianCallh/IterGP.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Usage 

IterGPs uses the [AbstractGPs](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) interface.


```julia
using Random
using LinearAlgebra
using Plots
using IterGPs
using AbstractGPs
using KernelFunctions

n = 500
σ² = 0.25
rng = MersenneTwister(1234)
x = Float64.(collect(range(-4, 4, n)))
y = sin.(x) .+ sqrt(σ²)*randn(rng, length(x))
xx = collect(1.4 .* range(extrema(x)..., 200))

function plot_fit(f, xx, σ²,x, y)
    fx = f(xx, σ²)
    plot(xx, mean(fx), ribbon=2 .* sqrt.(var(fx)), label="Model")
    scatter!(x, y, label="Data")
end

kernel = Matern32Kernel()

# Cholesky actions
chol_fx = chol_f(x, σ²)
chol_pf = posterior(chol_fx, y)
cholesky_fit_plt = plot_fit(chol_pf, xx, σ², x, y)

# Conjugate gradient actions
numiters = 100
x0 = zeros(length(x))
cg_f = IterGP(kernel, ConjugateGradientPolicy(x0, numiters))
cg_fx = cg_f(x, σ²)
cg_pf = posterior(cg_fx, y)
cg_fit_plt = plot_fit(cg_pf, xx, σ², x, y)
```

# Implementation details
The user facing API exposes three primitives: the `IterGP` constructor, various policites, e.g. `ConjugateGradientPolicy`, and the `CholeskyPreconditioner`. Having used these primitives to construct a `GP`, you can call `AbstractGPs.posterior` on it, which is where the magic happens.

As a way to cache computations (such as preconditioners), the policy object itself is not called in the innermost loop. Instead, the policy is used to create an /actor/ which is then called in the inner loop until a convergence criteria is met.

# References
Apart from the actual paper, these lecture notes ([Scaling GPs](https://media.githubusercontent.com/media/philipphennig/NumericsOfML/main/slides/03_ScalingGPs.pdf)), ([Computation aware GPs](
https://media.githubusercontent.com/media/philipphennig/NumericsOfML/main/slides/04_ComputationAwareGPs.pdf)) explain a lot of the implementation.


