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

function return_trajectories(policy, wrapper, num_trajectories)
    trajectories = Vector{Float64}[]
    for _ in 1:num_trajectories
        PPO.reset!(wrapper)
        ret = SQ.single_trajectory_normalized_return_history(policy, wrapper)
        push!(trajectories, ret)
    end
    return trajectories
end

function pad_trajectories!(trajectories)
    num_steps = length.(trajectories)
    max_steps = maximum(num_steps)
    for (idx, ret) in enumerate(trajectories)
        num_new_entries = max_steps - length(ret)
        padded_ret = SQ.QM.pad_vector(ret, num_new_entries, ret[end])
        trajectories[idx] = padded_ret
    end
end


input_dir = "output/poly-10-30-ent-2e-2/"
number_of_trajectories = 100
max_actions_factor = 4

data_filename = joinpath(input_dir, "best_model.bson")
data = BSON.load(data_filename)[:data];
policy = data["policy"]

config_file = joinpath(input_dir, "config.toml")
config = TOML.parsefile(config_file)

env_config = config["environment"]
env_config["max_actions_factor"] = max_actions_factor
wrapper = initialize_environment(env_config)

trajectories = return_trajectories(policy, wrapper, number_of_trajectories)
pad_trajectories!(trajectories)
concat_trajectories = cat(trajectories..., dims=2)
avg_trajectory = SQ.Flux.mean(concat_trajectories, dims=2)

fig, ax = subplots()
for ret in trajectories
    ax.plot(ret, color = "cornflowerblue", alpha=0.2)
end
ax.plot(avg_trajectory, color="royalblue", linewidth=3)
ax.set_xlabel("Number of steps")
ax.set_ylabel("Normalized returns")
ax.set_ylim([0,1])
ax.grid(true)
fig.tight_layout()
fig

output_file = joinpath(input_dir, "figures", "trajectory_history.pdf")
fig.savefig(output_file)