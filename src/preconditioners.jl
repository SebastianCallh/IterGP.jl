abstract type AbstractPreconditioner end

struct CholeskyPreconditioner <: AbstractPreconditioner
    rank::Int
end

function (p::CholeskyPreconditioner)(K, Σy)
    n, k = size(K, 1), p.rank
    K′ = copy(K)
    L = Array{eltype(K)}(undef, n, k)
    for i in 1:k
        Iᵢ = K′[:,i] / sqrt(K′[i,i])
        K′ .= K′ - Iᵢ*Iᵢ'
        L[:,i] = Iᵢ
    end
    L*L' + Σy
end

struct NoPreconditioner <: AbstractPreconditioner end
function (p::NoPreconditioner)(_, _) I end