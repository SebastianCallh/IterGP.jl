# Yarr! Neede due to https://github.com/timholy/WoodburyMatrices.jl/issues/35
function WoodburyMatrices._ldiv!(dest, W::SymWoodbury, A::Union{Factorization, Diagonal}, B)
    WoodburyMatrices.myldiv!(W.tmpN1, A, B)
    mul!(W.tmpk1, W.V, W.tmpN1)
    mul!(W.tmpk2, W.Cp, W.tmpk1)
    mul!(W.tmpN2, W.U, W.tmpk2)
    WoodburyMatrices.myldiv!(A, W.tmpN2)
    for i = 1:length(W.tmpN2)
        @inbounds dest[i] = W.tmpN1[i] - W.tmpN2[i]
    end
    return dest
end
