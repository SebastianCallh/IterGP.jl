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
    p.i += 1
    return e
end

function update!(p::CholeskyPolicy, dᵢ, αᵢ, ηᵢ)
    # do nothing    
end

function done(p::CholeskyPolicy, args...) 
    p.i > p.rank
end

mutable struct ConjugateGradientPolicy{T <: AbstractFloat} <: Policy
    A::Matrix{T}
    b::Vector{T}
    x::Vector{T}
    maxiters::Int64
    abstol::Float64
    reltol::Float64
end

function done(p::ConjugateGradientPolicy, r)
    norm(r) <= max(p.reltol*norm(p.b), p.abstol)
end

function action(p::ConjugateGradientPolicy)
    p.b - p.A*p.x
end

function update!(p::ConjugateGradientPolicy, dᵢ, αᵢ, ηᵢ)
    p.x += dᵢ * αᵢ/ηᵢ
end