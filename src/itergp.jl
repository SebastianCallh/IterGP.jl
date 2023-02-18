module IterGP

using LinearAlgebra
using KernelFunctions
using Distributions
using RecipesBase

export CholeskyPolicy, ConjugateGradientPolicy, action, done, update!
export GP, posterior

include("policies.jl")
include("gp.jl")

end