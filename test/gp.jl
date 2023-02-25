@testset begin
    

    n = 100
    σ² = 0.01
    rng = MersenneTwister(12345)
    x, y = sinusoid(rng, n, σ²)
    xx = collect(1.4 .* range(extrema(x)..., 200))
    maxiters = 500
    preconditioner = CholeskyPreconditioner(15)
    kernel = Matern32Kernel()

    pcg = ConjugateGradientPolicy(zeros(n), maxiters, P=preconditioner, abstol=1e-5, reltol=1e-5)
    pcg_f = IterGP(kernel, pcg)
    pcg_fx = pcg_f(x, σ²)
    pcg_pf, _ = posterior(pcg_fx, y)

    cg = ConjugateGradientPolicy(zeros(n), maxiters, abstol=1e-5, reltol=1e-5)
    cg_f = IterGP(kernel, cg)
    cg_fx = cg_f(x, σ²)
    cg_pf, _ = posterior(cg_fx, y)

    # Preconditioned CG converges to the same solution as vanilla
    @test isapprox(pcg_pf.v, cg_pf.v, rtol=1e-4, atol=1e-8)
    @test isapprox(pcg_pf.C, cg_pf.C, rtol=1e-4, atol=1e-8)
end