module IterGP

using LinearAlgebra
using KernelFunctions
using Distributions
using RecipesBase
using WoodburyMatrices

export CholeskyPolicy, ConjugateGradientPolicy, action, done, update!
export GP, posterior

include("policies.jl")
include("preconditioners.jl")
include("gp.jl")

end