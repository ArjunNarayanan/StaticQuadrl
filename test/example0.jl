using StaticQuadrl
using ProximalPolicyOptimization
PPO = ProximalPolicyOptimization
SQ = StaticQuadrl

include("plot.jl")
include("ppo_definitions.jl")


wrapper = SQ.RandPolyEnv(
    [20],
    3,
    "catmull-clark",
    true,
    true,
)
policy = SQ.SimplePolicy(216, 128, 5, SQ.NUM_ACTIONS_PER_EDGE)


rollouts = PPO.Rollouts("output/model-1")