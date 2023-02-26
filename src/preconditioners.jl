abstract type AbstractPreconditioner end

struct CholeskyPreconditioner{T <: AbstractFloat} <: AbstractPreconditioner
    rank::Int64
    jitter::T
    zero_threshold::T
end

CholeskyPreconditioner(rank, jitter=1e-8, zero_threshold=1e-12) = CholeskyPreconditioner(rank, jitter, zero_threshold)

function (p::CholeskyPreconditioner)(A, Σy)
    A′ = copy(A) + Diagonal(fill(p.jitter, size(A, 1)))
    n, k = size(A, 1), p.rank
    L = Array{eltype(A)}(undef, n, k)
    for i in 1:k
        Iᵢ = A′[:,i] / sqrt(A′[i,i])
        A′ .= A′ - Iᵢ*Iᵢ'
        L[:,i] = Iᵢ
    end

    D = Diagonal(ones(k))
    L[L .< p.zero_threshold] .= 0
    SymWoodbury(Σy, L, D)
end

struct DiagonalPreconditioner{T <: AbstractVector{<:AbstractFloat}} <: AbstractPreconditioner
    diagonal::T
end

function (p::DiagonalPreconditioner)(A, args...)
    p.diagonal*I(size(A, 1))
end