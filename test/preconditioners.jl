@testset "cholesky" begin
    n = 100
    x = collect(1:n)
    y = x + randn(n)
    K = kernelmatrix(RBFKernel(), x)
    σ²I = Diagonal(Fill(1, n))
    A = K + σ²I
    b = Zeros(n)

    ## Preconditioning does not change solution of linear system
    P = CholeskyPreconditioner(rand(1:n))(K, σ²I)
    @test isapprox(A\b, (P\A) \ (P\b))
end