module IterGPs

using LinearAlgebra
using Statistics
using Random
using StatsBase
using AbstractGPs
using KernelFunctions
using Distributions
using WoodburyMatrices
using RecipesBase
using FillArrays
using AbstractGPs: AbstractGP, FiniteGP, MeanFunction, ZeroMean
using AbstractGPs: _symmetric, _map_meanfunction

export IterGP, posterior, DiagonalPreconditioner, CholeskyPreconditioner, CholeskyPolicy, ConjugateGradientPolicy, sinusoid

include("piracy.jl")
include("datasets.jl")
include("preconditioners.jl")
include("policies.jl")
include("actors.jl")
include("gp.jl")
end