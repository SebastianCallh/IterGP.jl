@testset "cholesky" begin
    n = 100
    x = collect(1:n)
    y = x + randn(n)
    K = kernelmatrix(RBFKernel(), x)
    σ²I = Diagonal(Fill(1, n))
    A = K + σ²I
    b = Zeros(n)

    # Preconditioning does not change solution of linear system
    Pinv = inv(CholeskyPreconditioner(rand(10:n))(K, σ²I))
    @test isapprox(A\b, Pinv*A \ Pinv*b)

    # Applying preconditioner reduces condition number
    @test cond(Pinv*A) < cond(A)
end
