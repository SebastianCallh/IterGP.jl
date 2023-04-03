abstract type AbstractPolicy end

struct CholeskyPolicy <: AbstractPolicy
    dim::Int
    rank::Int
    i::Int
    CholeskyPolicy(dim, rank) = new(dim, rank, 1)
end

function actor(p::CholeskyPolicy, K, args...)
    CholeskyActor(size(K, 1), p.rank)
end

struct ConjugateGradientPolicy{
    P <: Union{Nothing, AbstractPreconditioner},
    T <: AbstractFloat,
    Tx <: AbstractVector{T}
} <: AbstractPolicy
    x₀::Tx
    Pinv::P
    maxiters::Int
    atol::T
    rtol::T
end

ConjugateGradientPolicy(x₀, maxiters; atol=1e-6, rtol=1e-16) = ConjugateGradientPolicy(x₀, NoPreconditioner(), maxiters, atol, rtol)
ConjugateGradientPolicy(x₀, maxiters, P; atol=1e-6, rtol=1e-16) = ConjugateGradientPolicy(x₀, P, maxiters, atol, rtol)

maxiters(p::ConjugateGradientPolicy) = p.maxiters

function actor(p::ConjugateGradientPolicy, K, Σy, δ)
    K̂ = K + Σy
    Pinv = inv(p.Pinv(K, Σy))
    ConjugateGradientActor(K̂, δ, copy(p.x₀), Pinv, p.maxiters, p.atol, p.rtol)
end