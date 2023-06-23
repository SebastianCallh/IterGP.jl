module IterGP

using Random
using Statistics
using LinearAlgebra
using StatsBase
using KernelFunctions
using RecipesBase
using AbstractGPs
using AbstractGPs: AbstractGP, FiniteGP, MeanFunction, ZeroMean

export CholeskyPolicy, ConjugateGradientPolicy
export NoPreconditioner, CholeskyPreconditioner

include("preconditioners.jl")
include("policies.jl")
include("actors.jl")
include("posterior.jl")

end
