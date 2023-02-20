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

mutable struct ConjugateGradientPolicy{T <: AbstractFloat} <: Policy
    A::Matrix{T}
    b::Vector{T}
    x::Vector{T}
    P::Union{Nothing, SymWoodbury{T}}
    maxiters::Int64
    abstol::Float64
    reltol::Float64
end

function done(p::ConjugateGradientPolicy, r, i)
    return (
        norm(r) <= max(p.reltol*norm(p.b), p.abstol) ||
        i >= p.maxiters
    )
end

function action(p::ConjugateGradientPolicy)
    (;A, b, x, P) = p
    r = b - A*x
    !isnothing(P) ? P\r : r
end

function update!(p::ConjugateGradientPolicy, dᵢ, αᵢ, ηᵢ)
    p.x += dᵢ * αᵢ/ηᵢ
end

function maxiters(p::ConjugateGradientPolicy)
    p.maxiters
end