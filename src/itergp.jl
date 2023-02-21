using LinearAlgebra
using KernelFunctions
using Distributions
using WoodburyMatrices
using RecipesBase

export CholeskyPolicy, ConjugateGradientPolicy, PreconditionedConjugateGradientPolicy, action, done, update!
export GP, posterior
export cholesky_preconditioner
export log_det_approx, trace_approx

include("datasets.jl")
include("policies.jl")
include("preconditioners.jl")
include("inference.jl")
include("gp.jl")
