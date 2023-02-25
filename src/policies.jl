abstract type AbstractPolicy end

mutable struct CholeskyPolicy <: AbstractPolicy
    dim::Int32
    rank::Int32
    i::Int32
    CholeskyPolicy(dim, rank) = new(dim, rank, 1)
end

function action(p::CholeskyPolicy)
    e = zeros(p.dim)
    e[p.i] = 1.
    return e
end

function update!(p::CholeskyPolicy, args...)
    p.i += 1
end

function done(p::CholeskyPolicy, args...)
    p.i > p.rank
end

function maxiters(p::CholeskyPolicy)
    p.rank
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

function actor(p::ConjugateGradientPolicy, K, Σy, μ, y)
    A = K + Σy
    b = μ - y
    P = isnothing(p.preconditioner) ? nothing : p.preconditioner(K, Σy)
    ConjugateGradientActor(A, b, copy(p.x₀), P, p.maxiters, p.abstol, p.reltol)
end