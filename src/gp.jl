abstract type AbstractGP end

@recipe function f(gp::AbstractGP, xx)
    py = gp(xx)
    legend --> :topleft

    @series begin
        ribbon --> 2 .* sqrt.(var(py))
        ribbonalpha --> 0.5
        linewidth --> 3
        label --> "Mean and 2σ"
        xx, mean(py)
    end

    @series begin
        color --> 1
        alpha --> 0.5
        label --> nothing
        xx, rand(py, 10)
    end
end

struct GP{K <: Kernel, T <: AbstractFloat} <: AbstractGP
    mean::Returns{T}
    kernel::K
    σ²::T
end

function (gp::GP)(x)
    (; mean, kernel) = gp
    μ₀ = mean.(x)
    Σ₀ = Hermitian(kernelmatrix(kernel, x))
    MvNormal(μ₀, Σ₀)
end

struct Posterior{K <: Kernel, T <: AbstractFloat} <: AbstractGP
    mean::Returns{T}
    kernel::K
    v::Vector{T}
    C::Matrix{T}
    X::Vector{T}
    σ²::T
end

function (p::Posterior)(x)
    (; mean, kernel, X, v, C) = p
    μₙ = mean.(x) + kernelmatrix(kernel, x, X)*v
    Σₙ = Hermitian(
        kernelmatrix(kernel, x) -
        kernelmatrix(kernel, x, X)*C*kernelmatrix(kernel, X, x) + 
        p.σ²*I # Diagonal(fill(jitter, length(x)))
    )
    MvNormal(μₙ, Σₙ)
end

function posterior(policy, prior, X, y)
    (; mean, kernel, σ²) = prior
    K̂ = kernelmatrix(kernel, X) + σ²*I
    μ = mean.(X)

    r = zeros(length(y))
    v = zeros(length(y))    
    C = zeros(size(K̂))
    for i in 1:maxiters(policy)
        sᵢ = action(policy)
        r = (y - μ) - K̂*v
        αᵢ = sᵢ'r
        dᵢ = (I - C*K̂)*sᵢ
        ηᵢ = sᵢ'K̂*dᵢ
        C .+= dᵢ*dᵢ' ./ ηᵢ
        v .+= dᵢ*αᵢ/ηᵢ
        if done(policy, r, i)
            println("Converged in $i iterations\nResidual norm: $(norm(r))")
            break
        end
        update!(policy, dᵢ, αᵢ, ηᵢ)
    end

    Posterior(mean, kernel, v, C, X, σ²)
end