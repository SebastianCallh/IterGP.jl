module IterGP

using LinearAlgebra
using KernelFunctions
using Distributions
using RecipesBase
using WoodburyMatrices

export CholeskyPolicy, ConjugateGradientPolicy, action, done, update!
export GP, posterior
export cholesky_preconditioner
export log_det_approx, trace_approx

include("policies.jl")
include("preconditioners.jl")
include("inference.jl")
include("gp.jl")

end