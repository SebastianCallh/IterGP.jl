function sinusoid(rng, n, σ²)
    x = Float64.(shuffle(collect(range(-4, 4, n))))
    y = sin.(x) .+ sqrt(σ²)*randn(rng, length(x))
    return x, y    
end
