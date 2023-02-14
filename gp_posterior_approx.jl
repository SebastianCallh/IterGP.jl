module M
using LinearAlgebra
using KernelFunctions
using Distributions

struct Cholesky
    rank::Int32
end

μ₀(m, x) = m .* ones(Float64, length(x))
Σ₀(k, x) = kernelmatrix(k, x)

prior(k, m, x) = MvNormal(μ₀(m, x), Hermitian(Σ₀(k, x)))

μₙ(k, m, X, y, x, C) = μ₀(m, x) + kernelmatrix(k, x, X)*C*(y - μ₀(m, X))
Σₙ(k, X, x, C) = Hermitian(kernelmatrix(k, x, x) - kernelmatrix(k, x, X)*C*kernelmatrix(k, X, x))

function _chol(A, n)
    A′ = copy(A)
    C = zeros(size(A))
    L = diagm(ones(size(A, 1)))
    for i in 1:n
        sᵢ = let
            e = zeros(size(A, 1))
            e[i] = 1.
            e
        end
        dᵢ = (I - C*A)*sᵢ
        ηᵢ = sᵢ'A*dᵢ
        Iᵢ = A′[:,i] / sqrt(A′[i,i])
        C .= C + dᵢ*dᵢ' ./ ηᵢ
        A′ .= A′ - Iᵢ*Iᵢ'
        L[:,i] = Iᵢ
    end
    LowerTriangular(L), C
end

function (c::Cholesky)(k, m, X, y, x, σ²)
    _, C = _chol(kernelmatrix(k, X, X) + σ²*I, c.rank)
    MvNormal(μₙ(k, m, X, y, x, C), Σₙ(k, X, x, C))
end

struct ConjugateGradient{T <: AbstractVector{<:AbstractFloat}}
    x0::T
    maxiters::Int64
    rtol::Float64
    atol::Float64
end

ConjugateGradient(x0) = ConjugateGradient(x0, size(x0, 1), 1e-6, 1e-6)

function (cg::ConjugateGradient)(k, m, X, y, x, σ²)
    A = kernelmatrix(k, X, X) + σ²*I
    b = y - μ₀(m, X)
    _, C = _cg(A, b, cg.x0, cg.maxiters, cg.rtol, cg.atol)
    MvNormal(μₙ(k, m, X, y, x, C), Σₙ(k, X, x, C))
end

function _cg(A, b, x₀, maxiters, rtol, atol)
    x = copy(x₀)
    C = zeros(size(A))
    r = fill(Inf, length(b))
    i = 0
    while norm(r) > max(rtol*norm(b), atol)
        if i == maxiters
            println("Maximum number of iterations reached")
            return x, C
        end
        i += 1

        r = b - A*x
        sᵢ = r
        αᵢ = sᵢ'r
        dᵢ = (I - C*A)*sᵢ
        ηᵢ = sᵢ'A*dᵢ
        C .= C + dᵢ*dᵢ' ./ ηᵢ
        x .+= dᵢ * αᵢ/ηᵢ
    end
    x, C
end

end

using KernelFunctions
using Plots
using Distributions

x = Float64.(collect(range(-1, 1, 25)))
xx = collect(1.2 .* range(extrema(x)..., 200))
y = sin.(x) .+ 0.15*randn(length(x))
k = Matern32Kernel()
m = 0
σ² = 0.15
heatmap(kernelmatrix(k, xx), yflip=true)

function plot_gp(xx, py, x, y)
    plot(xx, mean(py), ribbon=2 .* sqrt.(var(py)), ribbonalpha=0.5)
    scatter!(x, y, label="Data")
end
prior = M.prior(k, m, xx);
plot_gp(xx, prior, x, y)

strategy = M.Cholesky(10)
chol_posterior = strategy(k, m, x, y, xx, σ²);
plot_gp(xx, chol_posterior, x, y)

x0 = randn(size(x, 1))
strategy = M.ConjugateGradient(x0, 3, 1e-6, 1e-6)
cg_posterior = strategy(k, m, x, y, xx, σ²);
plot_gp(xx, cg_posterior, x, y)