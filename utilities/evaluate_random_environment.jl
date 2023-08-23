using BSON
using TOML
using CUDA
using ProximalPolicyOptimization
include("../src/StaticQuadrl.jl")
include("../src/plot.jl")
SQ = StaticQuadrl
PPO = ProximalPolicyOptimization

function initialize_environment(env_config)
    polygon_degree_list = env_config["min_polygon_degree"]:env_config["max_polygon_degree"]
    env = SQ.RandPolyEnv(
        polygon_degree_list,
        env_config["max_actions_factor"],
        env_config["quad_alg"],
        env_config["cleanup"],
        env_config["round_desired_degree"]
    )
    return env
end


input_dir = "output/poly-10-20/"
number_of_trajectories = 100
max_actions_factor = 4

data_filename = joinpath(input_dir, "best_model.bson")
data = BSON.load(data_filename)[:data];
policy = data["policy"]

config_file = joinpath(input_dir, "config.toml")
config = TOML.parsefile(config_file)

env_config = config["environment"]
env_config["max_actions_factor"] = max_actions_factor
env_config["min_polygon_degree"] = 30
env_config["max_polygon_degree"] = 30
wrapper = initialize_environment(env_config)

ret, dev = SQ.average_normalized_best_returns(policy, wrapper, number_of_trajectories)