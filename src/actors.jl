abstract type AbstractActor end

mutable struct CholeskyActor <: AbstractActor
    dim::Int
    rank::Int
    i::Int
    CholeskyActor(dim::Int, rank::Int) = new(dim, rank, 1)
end

action(a::CholeskyActor) = begin
    e = zeros(a.dim)
    e[a.i] = 1.
    return e
end

update!(a::CholeskyActor, args...) = begin
    a.i += 1
end

done(a::CholeskyActor, args...) = begin
    a.i > a.rank
end

struct ConjugateGradientActor{
    T <: AbstractFloat,
    TA<: AbstractMatrix{T},
    Tb <: AbstractVector{T},
    Tx <: AbstractVector{T},
    TP
}
    A::TA
    b::Tb
    x::Tx
    P::TP
    maxiters::Int
    abstol::T
    reltol::T
end

action(a::ConjugateGradientActor) = begin 
    a.P\(a.b - a.A*a.x)
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