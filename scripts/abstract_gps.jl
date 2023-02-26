using IterGPs
using Random
using LinearAlgebra
using KernelFunctions
using Plots
using AbstractGPs

n = 250
σ² = 0.1
rng = MersenneTwister(12345)
x = Float64.(shuffle(rng, collect(range(-3, 3, n))))
y = sin.(x) .+ sqrt(σ²)*randn(rng, length(x))
xx = collect(1.4 .* range(extrema(x)..., 200))
kernel = Matern52Kernel()
scatter(x, y)

# cholesky policy
chol_policy = CholeskyPolicy(length(x), n)
chol_f = IterGP(kernel, chol_policy)
chol_fx = chol_f(x, σ²)
chol_fxx = chol_f(xx, σ²)
chol_pf, rs = posterior(chol_fx, y);

chol_pfxx = chol_pf(xx, σ²)
chol_plt = plot(xx, mean(chol_pfxx), ribbon=2 .* sqrt.(var(chol_pfxx)), label="Cholesky policy ")
scatter!(chol_plt, x, y, label="Data", color=2)

# Preconditioned conjugate gradient
# Hits the convergence criteria, but posterior variance is too big
maxiters = n
P = CholeskyPreconditioner(15)
pcg = ConjugateGradientPolicy(zeros(n), maxiters, P; atol=1e-5, rtol=1e-5)
pcg_f = IterGP(kernel, pcg)
pcg_fx = pcg_f(x, σ²)
pcg_pf, rs = posterior(pcg_fx, y);
plot(norm.(rs))
pcg_pfxx = pcg_pf(xx, σ²);



## Vanilla conjugate gradients
## works as expected
cg = ConjugateGradientPolicy(zeros(n), maxiters; atol=1e-5, rtol=1e-5)
cg_f = IterGP(kernel, cg)
cg_fx = cg_f(x, σ²)
cg_pf, rs = posterior(cg_fx, y);
plot(norm.(rs))
cg_pfxx = cg_pf(xx, σ²);


## Preconditioned conjugat gradients but forced to run many more iterations than data points
## Produces the correct posterior variance for some reason
pcg_tol = ConjugateGradientPolicy(zeros(n), 2n, P; atol=1e-16, rtol=1e-16)
pcg_tol_f = IterGP(kernel, pcg_tol)
pcg_tol_fx = pcg_tol_f(x, σ²)
pcg_tol_pf, rs = posterior(pcg_tol_fx, y);
plot(norm.(rs))
pcg_tol_pfxx = pcg_tol_pf(xx, σ²);


isapprox(pcg_pf.v, cg_pf.v, rtol=1e-4, atol=1e-8)
isapprox(pcg_pf.C, cg_pf.C; rtol=1e-2, atol=1e-2)
isapprox(pcg_tol_pf.v, cg_pf.v, rtol=1e-4, atol=1e-8)
isapprox(pcg_tol_pf.C, cg_pf.C; rtol=1e-1, atol=1e-1)

# Reference implementation
f = GP(kernel)
fx = f(x, σ²)
pf = posterior(fx, y)
pfxx = pf(xx, σ²)

fit_plt = plot(xx, mean(pcg_pfxx), ribbon=2 .* sqrt.(var(pcg_pfxx)), label="IterGP PCG")
plot!(fit_plt, xx, mean(cg_pfxx), ribbon=2 .* sqrt.(var(cg_pfxx)), label="IterGP CG")
plot!(fit_plt, xx, mean(pcg_tol_pfxx), ribbon=2 .* sqrt.(var(pcg_tol_pfxx)), label="IterGP PCG maxiter")
plot!(fit_plt, xx, mean(pfxx), ribbon=2 .* sqrt.(var(pfxx)), label="GP")
scatter!(fit_plt, x, y, label="Data", color=2)
savefig(fit_plt, joinpath("plots", "abstract_gps_fits.png"))