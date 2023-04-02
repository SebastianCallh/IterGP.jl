function Distributions.logpdf(fx::FiniteGP{<:IterGP}, y::AbstractVecOrMat{<:Real})
    δ = (y - mean(fx))
    K = cov(fx)
    n = lengt(δ)

    P = nothing # currently we do not know if the GP `fx` has a preconditioner
    preconditioner_rank = nothing
    pz̃ = MvNormal(zeros(n), I)
    zs = rand(pz̃, preconditioner_rank) ./ sqrt(n)
    
    -0.5*(δ'K\δ - log_det_approx(K, P, zs) -  n*log(2π))
end

function trace_approx(A, zs)
    n = size(A, 1)
    ℓ = size(zs, 2)
    mapreduce(z -> n/ℓ * z'A*z, +, eachcol(zs))
end

function log_det_approx(K, P, zs)
    logdet(P) + trace_approx(log(K) - log(P), zs)
end