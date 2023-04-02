struct IterGP{M<:MeanFunction, K<:Kernel, P<:AbstractPolicy} <: AbstractGP
    mean::M
    kernel::K
    policy::P
end

IterGP(mean, kernel::Kernel, policy::AbstractPolicy) = IterGP(CustomMean(mean), kernel, policy)
IterGP(mean::Real, kernel::Kernel, policy::AbstractPolicy) = IterGP(ConstMean(mean), kernel, policy)
IterGP(kernel::Kernel, policy::AbstractPolicy) = IterGP(ZeroMean(), kernel, policy)

Statistics.mean(fx::FiniteGP{<:IterGP}) = _map_meanfunction(fx.f.mean, fx.x)
Statistics.cov(fx::FiniteGP{<:IterGP}) = kernelmatrix(fx.f.kernel, fx.x) + fx.Σy
Statistics.var(fx::FiniteGP{<:IterGP}) = kernelmatrix_diag(fx.f.kernel, fx.x) + diag(fx.Σy)

function Random.rand(rng::AbstractRNG, fx::FiniteGP{<:IterGP}, N::Int = 1)
    m, C_mat = mean_and_cov(fx)
    C = cholesky(Hermitian(_symmetric(C_mat)))
    return m .+ C.U' * randn(rng, promote_type(eltype(m), eltype(C)), length(m), N)    
end

@recipe function f(gp::FiniteGP{<:IterGP}, xx)
    py = gp(xx)
    legend --> :topleft
    title --> "Prior"

    @series begin
        ribbon --> 2 .* sqrt.(var(py))
        ribbonalpha --> 0.5
        linewidth --> 3
        label --> "Mean and 2σ"
        xx, mean(py)
    end

    @series begin
        color --> 1
        alpha --> 0.5
        label --> nothing
        xx, rand(py, 10)
    end
end

struct PosteriorIterGP{
    Tprior,
    T <: AbstractFloat,
    Ty <: AbstractVector{T},
    Tv <: AbstractVector{T},
    TC <: AbstractMatrix{T}
} <: AbstractGP
    prior::Tprior
    y::Ty
    v::Tv
    C::TC
end

Statistics.mean(pfx::FiniteGP{<:PosteriorIterGP}) = begin
    m = pfx.f.prior.f.mean
    K = pfx.f.prior.f.kernel
    X = pfx.f.prior.x
    x = pfx.x
    v = pfx.f.v
    _map_meanfunction(m, x) + kernelmatrix(K, x, X)*v
end

Statistics.var(pfx::FiniteGP{<:PosteriorIterGP}) = diag(cov(pfx)) # should be smarter
Statistics.cov(pfx::FiniteGP{<:PosteriorIterGP}) = begin
    K = pfx.f.prior.f.kernel
    C = pfx.f.C
    X = pfx.f.prior.x
    x = pfx.x
    Σ = kernelmatrix(K, x) - kernelmatrix(K, x, X)*C*kernelmatrix(K, X, x)
    Σ + pfx.Σy
end

StatsBase.mean_and_cov(pfx::FiniteGP{<:PosteriorIterGP}) = begin
    (;mean, kernel) = pfx.f.prior.f
    (;v, C) = (pfx.f)    
    X = pfx.f.prior.x
    x = pfx.x

    kxX = kernelmatrix(kernel, x, X)
    μ = _map_meanfunction(mean, x) + kxX*v
    Σ = kernelmatrix(kernel, x) - kxX*C*kernelmatrix(kernel, X, x)
    μ, Σ + pfx.Σy
end

function Random.rand(rng::AbstractRNG, pfx::FiniteGP{<:PosteriorIterGP}, N::Int = 1)
    μ, Σ = mean_and_cov(pfx)
    C = cholesky(Hermitian(_symmetric(Σ)))
    return μ .+ C.U' * randn(rng, promote_type(eltype(μ), eltype(C)), length(μ), N)
end

function AbstractGPs.posterior(fx::FiniteGP{<:IterGP}, y::AbstractVector{<:Real})
    K = kernelmatrix(fx.f.kernel, fx.x)
    K̂ = K + fx.Σy
    δ = y - mean(fx)
    act = actor(fx.f.policy, K, fx.Σy, δ)
    r = fill(Inf, length(y))
    v = zeros(length(y))
    C = zeros(size(K̂))
    
    i = 0
    while !done(act, r, i)
        sᵢ = action(act)
        r = δ - K̂*v
        αᵢ = sᵢ'r
        dᵢ = (I - C*K̂)*sᵢ
        ηᵢ = sᵢ'K̂*dᵢ
        C .+= (1/ηᵢ)*dᵢ*dᵢ'
        v .+= dᵢ*αᵢ/ηᵢ
        update!(act, dᵢ, αᵢ, ηᵢ)
        i += 1
    end
    PosteriorIterGP(fx, y, v, C)
end
