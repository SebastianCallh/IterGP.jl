# IterGPs

A Julia implementation of [Posterior and Computational Uncertainty
in Gaussian Processes](https://arxiv.org/pdf/2205.15449.pdf).

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