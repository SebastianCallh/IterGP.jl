module IterGPs

using Random
using Statistics
using LinearAlgebra
using StatsBase
using KernelFunctions
using RecipesBase
using AbstractGPs
using AbstractGPs: AbstractGP, FiniteGP, MeanFunction, ZeroMean
using AbstractGPs: _symmetric, _map_meanfunction

export IterGP
export CholeskyPolicy, ConjugateGradientPolicy
export NoPreconditioner, CholeskyPreconditioner

include("preconditioners.jl")
include("policies.jl")
include("actors.jl")
include("itergp.jl")

end