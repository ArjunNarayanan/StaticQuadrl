using BSON
using TOML
using CUDA
using ProximalPolicyOptimization
include("../src/StaticQuadrl.jl")
include("../src/plot.jl")
SQ = StaticQuadrl
PPO = ProximalPolicyOptimization

function initialize_fixed_environment(env_config)
    min_poly_degree = env_config["min_polygon_degree"]
    max_polyg_degree = env_config["max_polygon_degree"]
    polygon_degree = rand(min_poly_degree:max_polyg_degree)
    max_actions = env_config["max_actions_factor"]*polygon_degree
    mesh, d0 = SQ.initialize_random_mesh(
        polygon_degree,
        env_config["quad_alg"],
        env_config["round_desired_degree"]
    )
    cleanup = env_config["cleanup"]
    env = SQ.FixedMeshEnv(mesh, d0, max_actions, cleanup)
    return env
end

function best_return_from_rollouts(policy, wrapper, num_samples)
    best_return = -Inf
    for _ in 1:num_samples
        PPO.reset!(wrapper)
        ret = SQ.best_normalized_single_trajectory_return(policy, wrapper)
        best_return = max(best_return, ret)
    end
    return best_return
end

function average_best_returns(policy, env_config, num_samples, num_trajectories)
    ret = zeros(num_trajectories)
    for idx in 1:num_trajectories
        println("Evaluating trajectory : ", idx)
        wrapper = initialize_fixed_environment(env_config)
        ret[idx] = best_return_from_rollouts(policy, wrapper, num_samples)
    end
    return SQ.Flux.mean(ret), SQ.Flux.std(ret)
end

input_dir = "output/poly-10-20/"
number_of_trajectories = 10
max_actions_factor = 5
min_polygon_degree = 40
max_polygon_degree = 50

data_filename = joinpath(input_dir, "best_model.bson")
data = BSON.load(data_filename)[:data];
policy = data["policy"]

config_file = joinpath(input_dir, "config.toml")
config = TOML.parsefile(config_file)

env_config = config["environment"]
env_config["max_actions_factor"] = max_actions_factor
env_config["min_polygon_degree"] = min_polygon_degree
env_config["max_polygon_degree"] = max_polygon_degree

ret, dev = average_best_returns(policy, env_config, 10, number_of_trajectories)
println("Ret ", ret)
println("Dev ", dev)