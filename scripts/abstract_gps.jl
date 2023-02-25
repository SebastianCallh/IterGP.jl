using IterGPs
using Random
using LinearAlgebra
using KernelFunctions
using Plots
using AbstractGPs

n = 500
σ² = 0.25
rng = MersenneTwister(12345)
x, y = sinusoid(rng, n, σ²)
xx = collect(1.4 .* range(extrema(x)..., 200))
maxiters = 500
kernel = Matern52Kernel()

# cholesky policy
chol_policy = CholeskyPolicy(length(x), n)
chol_f = IterGP(kernel, chol_policy)
chol_fx = chol_f(x, σ²)
chol_fxx = chol_f(xx, σ²)
chol_pf, rs = posterior(chol_fx, y);

chol_pfxx = chol_pf(xx, σ²)
chol_plt = plot(xx, mean(chol_pfxx), ribbon=2 .* sqrt.(var(chol_pfxx)), label="Cholesky policy ")
scatter!(chol_plt, x, y, label="Data", color=2)

# CG implementation
maxiters = 500
chol = CholeskyPreconditioner(15)
policy = ConjugateGradientPolicy(zeros(n), maxiters, P=chol, abstol=1e-5, reltol=1e-5)
kernel = Matern52Kernel()
f = IterGP(kernel, policy)
fx = f(x, σ²)
fxx = f(xx, σ²)


#= prior_plt = scatter(x, y, label="Data", title="Prior plot")
plot!(prior_plt, xx, mean(fxx), ribbon=2 .* sqrt.(var(fxx)), color=1, fillalpha=0.4)
=#

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



### compare with cholesky from main

using WoodburyMatrices
function cholesky_preconditioner(A, rank, S, zero_threshold=1e-8)
    A′ = copy(A)
    n, k = size(A, 1), rank
    L = Array{eltype(A)}(undef, n, k)
    for i in 1:rank
        Iᵢ = A′[:,i] / sqrt(A′[i,i])
        A′ .= A′ - Iᵢ*Iᵢ'
        L[:,i] = Iᵢ
    end

    D = Diagonal(ones(k))
    L[L .< zero_threshold] .= 0
    SymWoodbury(S, L, D)
end
Σy = pf(x, σ²).Σy
K = kernelmatrix(kernel, x) + Σy
rank = 15
Pold = cholesky_preconditioner(K, rank, Σy)
Pnew = CholeskyPreconditioner(rank)(K, Σy)
isapprox(Matrix(Pold), Matrix(Pnew))
