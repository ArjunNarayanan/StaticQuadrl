using BSON
using TOML
using CUDA
using ProximalPolicyOptimization
include("../src/StaticQuadrl.jl")
include("../src/plot.jl")
SQ = StaticQuadrl
PPO = ProximalPolicyOptimization

function initialize_polygon_environment(
    polygon_config, 
    max_actions, 
    allow_vertex_insert,
    hmax,
    cleanup=true,
)
    boundary = polygon_config["boundary"]
    boundary = Float64.(vcat(boundary[1]', boundary[2]'))
    mesh, d0 = SQ.generate_mesh_from_boundary(
        boundary,
        "catmull-clark",
        true,
        hmax = hmax,
        allow_vertex_insert = allow_vertex_insert
    )
    env = SQ.FixedMeshEnv(mesh, d0, max_actions, cleanup)
    return env
end

model_checkpoint = "output/poly-10-20/best_model.bson"
input_dir = "polygons/double-notch"
polygon_config_file = joinpath(input_dir, "double-notch.toml")
polygon_config = TOML.parsefile(polygon_config_file)
number_of_trajectories = 100
max_actions = 50
hmax = 0.6
allow_vertex_insert = true

wrapper = initialize_polygon_environment(polygon_config, max_actions, allow_vertex_insert, hmax)

PPO.reset!(wrapper)
fig = plot_wrapper(
    wrapper,
    ylim=[-0.1,1.1],
    xlim=[-0.1,1.1],
    plot_score=false
)
fig.savefig("polygons/double-notch/figures/initial.pdf")

data = BSON.load(model_checkpoint)[:data];
policy = data["policy"]

ret = SQ.best_normalized_return(policy, wrapper, number_of_trajectories)

best_wrapper = SQ.best_state_in_rollouts(wrapper, policy, number_of_trajectories)

fig = plot_wrapper(
    best_wrapper,
    ylim=[-0.1,1.1],
    xlim=[-0.1,1.1],
    plot_score=false
)
fig
fig.savefig("polygons/double-notch/figures/optimal.pdf")

rollout = 1
output_dir = joinpath(input_dir, "rollout-"*string(rollout))
PPO.reset!(wrapper)
plot_trajectory(
    policy, 
    wrapper, 
    output_dir,
    ylim=[-0.1,1.1],
    xlim=[-0.1,1.1],
    plot_score=false,
    extension = ".pdf"
)

# plot_wrapper(
#     wrapper,
#     ylim=[-0.5,2.5],
#     xlim=[-0.5,2.5],
#     plot_score=false
# )

# smooth_non_geometric_boundary_vertices!(wrapper.env.mesh)

