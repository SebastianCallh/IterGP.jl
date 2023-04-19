struct IterGP{
    Tprior<:FiniteGP,
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

Statistics.mean(f::IterGP, x::AbstractVector) = begin
    mean(f.prior.f, x) + cov(f.prior.f, x, f.prior.x)*f.v
end

Statistics.var(f::IterGP, x::AbstractVector) = diag(cov(f, x)) # should be smarter
Statistics.cov(f::IterGP, x::AbstractVector) = begin
    cov(f.prior.f, x) - cov(f.prior.f, x, f.prior.x)*f.C*cov(f.prior.f, f.prior.x, x)    
end

StatsBase.mean_and_cov(f::IterGP, x::AbstractVector) = begin
    mean(f, x), cov(f, x)
end

StatsBase.mean_and_var(f::IterGP, x::AbstractVector) = begin
    mean(f, x), var(f, x)
end

function Random.rand(rng::AbstractRNG, pfx::FiniteGP{<:IterGP}, N::Int = 1)
    μ, Σ = mean_and_cov(pfx)
    C = cholesky(Hermitian(Σ))
    return μ .+ C.U' * randn(rng, promote_type(eltype(μ), eltype(C)), length(μ), N)
end

function AbstractGPs.posterior(fx::FiniteGP, y::AbstractVector{<:Real}, policy::AbstractPolicy)
    K = kernelmatrix(fx.f.kernel, fx.x)
    K̂ = K + fx.Σy
    δ = y - mean(fx)
    act = actor(policy, K, fx.Σy, δ)
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
    IterGP(fx, y, v, C)
end
