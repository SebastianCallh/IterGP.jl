@testset begin
    n = 100
    σ = 0.1
    rng = MersenneTwister(12345)
    x = Float64.(collect(range(-4, 4, n)))
    y = sin.(x) .+ σ*randn(rng, length(x))
    xx = collect(1.4 .* range(extrema(x)..., 200))
    maxiters = 100
    preconditioner = CholeskyPreconditioner(15)
    kernel = Matern32Kernel()

    pcg = ConjugateGradientPolicy(zeros(n), maxiters, preconditioner)
    pcg_f = IterGP(kernel, pcg)
    pcg_fx = pcg_f(x, σ^2)
    pcg_pf = posterior(pcg_fx, y)

    cg = ConjugateGradientPolicy(zeros(n), maxiters)
    cg_f = IterGP(kernel, cg)
    cg_fx = cg_f(x, σ^2)
    cg_pf = posterior(cg_fx, y)

    # Preconditioned CG converges to the same solution as vanilla CG
    @test isapprox(pcg_pf.v, cg_pf.v, rtol=1e-5, atol=1e-5)
    @test isapprox(pcg_pf.C, cg_pf.C, rtol=1e-5, atol=1e-5)
end