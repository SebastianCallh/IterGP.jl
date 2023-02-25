using IterGPs
using Random
using LinearAlgebra
using KernelFunctions
using Plots
using AbstractGPs
#= 

using Preconditioners

using FillArrays
=#

n = 1000
σ² = 0.01
rng = MersenneTwister(12345)
x, y = sinusoid(rng, n, σ²)
xx = collect(1.4 .* range(extrema(x)..., 200))
# scatter(x, y)

# IterGP implementation
maxiters = 500
chol = CholeskyPreconditioner(15)
policy = ConjugateGradientPolicy(zeros(n), maxiters, P=chol, abstol=1e-5, reltol=1e-5)
kernel = Matern52Kernel()
f = IterGP(kernel, policy)
fx = f(x, σ²)
pf, rs = posterior(fx, y);
plot(norm.(rs))
pfxx = pf(xx, σ²);
pfx = pf(x, σ²);
K = kernelmatrix(kernel, x) + pf(x, σ²).Σy
Kinv = inv(K)

diag = DiagonalPreconditioner(1.)
policy2 = ConjugateGradientPolicy(zeros(n), maxiters, P=diag, abstol=1e-5, reltol=1e-5)
f2 = IterGP(kernel, policy2)
fx2 = f2(x, σ²)
pf2, rs = posterior(fx2, y);
plot(norm.(rs))
pfxx2 = pf2(xx, σ²);
pfx2 = pf2(x, σ²);
K2 = kernelmatrix(kernel, x) + pf2(x, σ²).Σy
K2inv = inv(K2)

isapprox(pf.v, pf2.v, rtol=1e-4, atol=1e-8)
isapprox(pfx2.f.C, pfx.f.C; rtol=1e-2, atol=1e-2)
isapprox(pfx.f.C, Kinv; rtol=1e-2, atol=1e-2)
isapprox(pfx.f.C, K2inv; rtol=1e-2, atol=1e-2)

pfxx.f.C - Kinv
pfxx.f.C - Kinv



#= 
heatmap(pfx.f.C)
heatmap(pfx2.f.C)
=#

# Reference implementation
f3 = GP(kernel)
fx3 = f3(x, σ²)
pf3 = posterior(fx3, y)
pfx3 = pf3(xx, σ²)

fit_plt1 = plot(xx, mean(pfxx), ribbon=2 .* sqrt.(var(pfxx)), label="IterGP cg precond")
plot!(xx, mean(pfxx2), ribbon=2 .* sqrt.(var(pfxx2)), label="IterGP cg")
plot!(xx, mean(pfx3), ribbon=2 .* sqrt.(var(pfx3)), label="GP vanilla")
scatter!(fit_plt1, x, y, label="Data", color=2)

fit_plt2 = plot(xx, rand(pfxx, 10), color=1, alpha=0.5, label=nothing)
plot!(fit_plt2, xx, rand(pfxx2, 10), color=2, alpha=0.5, label=nothing)
scatter!(fit_plt2, x, y, label="Data", color=3)

#= 
using Preconditioners
A = kernelmatrix(kernel, x) + σ²*I
pre = Preconditioners.CholeskyPreconditioner(A, 3)
pre.ldlt.D =#

K = kernelmatrix(kernel, x)
A = K + σ²*I
pre = preconditioner(K, Diagonal(fill(σ², length(x))))
P = pre.B*pre.B' + σ²*I
P2 = Matrix(pre)
P ≈ P2
cond(A)
cond(P\A)
cond(P2\A)
