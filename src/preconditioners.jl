function cholesky_preconditioner(A, rank, σ²)
    A′ = copy(A)
    n, k = size(A, 1), rank
    L = Array{eltype(A)}(undef, n, k)
    for i in 1:rank
        Iᵢ = A′[:,i] / sqrt(A′[i,i])
        A′ .= A′ - Iᵢ*Iᵢ'
        L[:,i] = Iᵢ
    end

    D = diagm(fill(σ², n))
    I = diagm(ones(k))
    SymWoodbury(D, L, I)
end