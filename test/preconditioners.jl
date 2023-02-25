@testset "cholesky" begin
    n = 100
    U = rand(n)
    K = U*U'
    D = Diagonal(Fill(1e-8, n))
    A = K + D
    b = Zeros(n)
   
    ## Preconditioning does not change solution of linear system
    P = CholeskyPreconditioner(rand(1:n))(K, D)
    @test isapprox(A\b, (P\A) \ (P\b))

    ## Preconditioning improves condition number
    @test cond(P\A) < cond(A)    
end