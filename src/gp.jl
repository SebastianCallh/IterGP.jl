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
    (; mean, kernel, σ²) = gp
    μ₀ = mean.(x)
    Σ₀ = Hermitian(kernelmatrix(kernel, x) + σ²*I) 
    MvNormal(μ₀, Σ₀)
end

struct Posterior{K <: Kernel, T <: AbstractFloat} <: AbstractGP
    mean::Returns{T}
    kernel::K
    v::Vector{T}
    C::Matrix{T}
    X::Vector{T}
end

function (p::Posterior)(x; jitter=1e-6)
    μₙ = p.mean.(x) + kernelmatrix(p.kernel, x, p.X)*p.v
    Σₙ = Hermitian(
        kernelmatrix(p.kernel, x) -
        kernelmatrix(p.kernel, x, p.X)*p.C*kernelmatrix(p.kernel, p.X, x) + 
        Diagonal(fill(jitter, length(x)))
    )
    MvNormal(μₙ, Σₙ)
end

function posterior(policy, prior, X, y)
    (; mean, kernel, σ²) = prior
    K̂ = kernelmatrix(kernel, X, X) + σ²*I
    r = fill(Inf, length(y))
    v = zeros(length(y))
    μ = mean.(X)
    C = zeros(size(K̂))
    while !done(policy, r)
        sᵢ = action(policy)
        r .= (y - μ) - K̂*v
        αᵢ = sᵢ'r
        dᵢ = (I - C*K̂)*sᵢ
        ηᵢ = sᵢ'K̂*dᵢ
        C .= C + dᵢ*dᵢ' ./ ηᵢ
        v .= v + dᵢ*αᵢ/ηᵢ
        update!(policy, dᵢ, αᵢ, ηᵢ)
    end

    Posterior(mean, kernel, v, C, X)
end