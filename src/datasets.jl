function sinusoid(rng, n, σ)    
    x = Float64.(collect(range(-4, 4, n)))
    y = sin.(x) .+ σ*randn(rng, length(x))
    return x, y    
end