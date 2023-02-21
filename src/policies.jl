abstract type Policy end

mutable struct CholeskyPolicy <: Policy
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


mutable struct PreconditionedConjugateGradientPolicy{T <: AbstractFloat} <: Policy
    A::Matrix{T}
    b::Vector{T}
    x::Vector{T}
    maxiters::Int64
    P::Union{SymWoodbury{T}, Matrix{T}}
    abstol::Float64
    reltol::Float64
end

function PreconditionedConjugateGradientPolicy(A, b, x, maxiters, P; abstol=1e-6, reltol=1e-6)
    PreconditionedConjugateGradientPolicy(A, b, x, maxiters, P, abstol, reltol)
end

function done(p::PreconditionedConjugateGradientPolicy, r, i)
    return (
        norm(r) <= max(p.reltol*norm(p.b), p.abstol) ||
        i >= p.maxiters
    )
end

function action(p::PreconditionedConjugateGradientPolicy)
    (;A, b, x, P) = p
    P\(b - A*x)
end

function update!(p::PreconditionedConjugateGradientPolicy, dᵢ, αᵢ, ηᵢ)
    p.x += dᵢ * αᵢ/ηᵢ
end

function maxiters(p::PreconditionedConjugateGradientPolicy)
    p.maxiters
end

mutable struct ConjugateGradientPolicy{T <: AbstractFloat} <: Policy
    A::Matrix{T}
    b::Vector{T}
    x::Vector{T}
    maxiters::Int64
    abstol::Float64
    reltol::Float64
end

function ConjugateGradientPolicy(A, b, x, maxiters; abstol=1e-6, reltol=1e-6)
    ConjugateGradientPolicy(A, b, x, maxiters, abstol, reltol)
end

function done(p::ConjugateGradientPolicy, r, i)
    return (
        norm(r) <= max(p.reltol*norm(p.b), p.abstol) ||
        i >= p.maxiters
    )
end

function action(p::ConjugateGradientPolicy)
    (;A, b, x) = p
    b - A*x
end

function update!(p::ConjugateGradientPolicy, dᵢ, αᵢ, ηᵢ)
    p.x += dᵢ * αᵢ/ηᵢ
end

function maxiters(p::ConjugateGradientPolicy)
    p.maxiters
end