module StaticQuadrl

using Flux
using RandomQuadMesh
using QuadMeshGame
using Distributions: Categorical
using BSON
using Printf
using ProximalPolicyOptimization

RQ = RandomQuadMesh
QM = QuadMeshGame
PPO = ProximalPolicyOptimization

const NUM_ACTIONS_PER_EDGE = 5
const NUM_EDGES_PER_ELEMENT = 4
const NO_ACTION_REWARD = -1
const NUM_EDGES_PER_TEMPLATE = 108

include("env_utilities.jl")
include("random_polygon_environment.jl")
include("policy.jl")
include("state.jl")
include("action_probabilities.jl")
include("step.jl")
include("evaluate.jl")

end
