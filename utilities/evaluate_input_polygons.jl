using BSON
using TOML
using CUDA
using ProximalPolicyOptimization
include("../src/StaticQuadrl.jl")
include("../src/plot.jl")
SQ = StaticQuadrl
PPO = ProximalPolicyOptimization

function initialize_polygon_environment(polygon_config, max_actions, cleanup=true)
    boundary = polygon_config["boundary"]
    boundary = Float64.(vcat(boundary[1]', boundary[2]'))
    mesh, d0 = SQ.generate_mesh_from_boundary(
        boundary,
        "catmull-clark",
        true
    )
    env = SQ.FixedMeshEnv(mesh, d0, max_actions, cleanup)
    return env
end

model_checkpoint = "output/poly-10-20/best_model.bson"
input_dir = "polygons/L-domain"
polygon_config_file = joinpath(input_dir, "L-domain.toml")
polygon_config = TOML.parsefile(polygon_config_file)
# number_of_trajectories = 100
max_actions = 20

wrapper = initialize_polygon_environment(polygon_config, max_actions)
PPO.reset!(wrapper)


plot_wrapper(
    wrapper,
    ylim=[-0.5,2.5],
    xlim=[-0.5,2.5],
    plot_score=false
)

data = BSON.load(model_checkpoint)[:data];
policy = data["policy"]

rollout = 2
output_dir = joinpath(input_dir, "rollout-"*string(rollout))
PPO.reset!(wrapper)
plot_trajectory(
    policy, 
    wrapper, 
    output_dir,
    ylim=[-0.5,2.5],
    xlim=[-0.5,2.5],
    plot_score=false
)

plot_wrapper(
    wrapper,
    ylim=[-0.5,2.5],
    xlim=[-0.5,2.5],
    plot_score=false
)

vertices = smooth_non_geometric_boundary_vertices!(wrapper.env.mesh)

# ret, dev = SQ.average_normalized_best_returns(policy, wrapper, number_of_trajectories)