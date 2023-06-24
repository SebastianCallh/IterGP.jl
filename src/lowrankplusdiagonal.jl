"""
Type for matrices on the form A = D + UU' that can be inverted using the matrix inversion lemma.
"""
struct LowRankPlusDiagonal{TD <: AbstractMatrix, TU <: AbstractMatrix}
    D::TD # Diagonal component
    U::TU # Low-rank component
end

function LinearAlgebra.inv(A::LowRankPlusDiagonal)
    (;D, U) = A
    V = U'
    Dinv = inv(D)
    B = I + V*Dinv*U
    Binv = inv(B)
    Dinv - Dinv*U*Binv*V*Dinv
end
