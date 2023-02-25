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
    preconditioner::P
    maxiters::Int
    abstol::T
    reltol::T
end

function ConjugateGradientPolicy(x₀, maxiters; P=nothing, abstol=1e-2, reltol=1e-2)
    ConjugateGradientPolicy(x₀, P, maxiters, abstol, reltol)
end

maxiters(p::ConjugateGradientPolicy) = p.maxiters

function actor(p::ConjugateGradientPolicy, K, Σy, δ)
    K̂ = K + Σy
    P = p.preconditioner(K, Σy)
    ConjugateGradientActor(K̂, δ, copy(p.x₀), P, p.maxiters, p.abstol, p.reltol)
end