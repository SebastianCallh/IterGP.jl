abstract type AbstractActor end

struct ConjugateGradientActor{
    T <: AbstractFloat,
    TA<: AbstractMatrix{T},
    Tb <: AbstractVector{T},
    Tx <: AbstractVector{T},
    TP <: Union{Nothing, SymWoodbury{T}, AbstractMatrix}
}
    A::TA
    b::Tb
    x::Tx
    P::TP
    maxiters::Int64
    abstol::T
    reltol::T
end

action(a::ConjugateGradientActor) = begin 
    r = a.b - a.A*a.x
    isnothing(a.P) ? r : a.P\r
end
update!(a::ConjugateGradientActor, dᵢ, αᵢ, ηᵢ) = a.x .+= dᵢ * αᵢ/ηᵢ
done(a::ConjugateGradientActor, r, i) = begin
    if i >= a.maxiters
        println("Maximum number of iterations ($i) reached.\nResidual norm: $(norm(r))")
        return true
    end

    if norm(r) <= max(a.reltol*norm(a.b), a.abstol)
        println("Converged in $i iterations\nResidual norm: $(norm(r))")
        return true
    end

    false
end

#= struct PreconditionedConjugateGradientActor{
    T <: AbstractFloat,
    TA<: AbstractMatrix{T},
    Tb <: AbstractVector{T},
    Tx <: AbstractVector{T},
    TP <: SymWoodbury{T}
}
    A::TA
    b::Tb
    x::Tx
    P::TP
    maxiters::Int64
    abstol::T
    reltol::T
end

action(a::PreconditionedConjugateGradientActor) = begin
    r = a.b - a.A*a.x
    s = a.P\r
    return s
end 
update!(a::PreconditionedConjugateGradientActor, dᵢ, αᵢ, ηᵢ) = a.x .+= dᵢ * αᵢ/ηᵢ
done(a::PreconditionedConjugateGradientActor, r, i) = begin
    if i >= a.maxiters
        println("Maximum number of iterations ($i) reached.\nResidual norm: $(norm(r))")
        return true
    end

    if norm(r) <= max(a.reltol*norm(a.b), a.abstol)
        println("Converged in $i iterations\nResidual norm: $(norm(r))")
        return true
    end

    false
end =#