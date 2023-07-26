module StaticQuadrl

using Flux
using RandomQuadMesh
using QuadMeshGame
using Distributions: Categorical
using BSON
using Printf

# using Revise
using ProximalPolicyOptimization

include("random_polygon_environment.jl")
include("state.jl")

end
