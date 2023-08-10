using BSON
include("../src/StaticQuadrl.jl")
using PyPlot


model_name = ARGS[1]
input_dir = joinpath("output", model_name)
output_dir = joinpath(input_dir, "figures")

if !isdir(output_dir)
    mkpath(output_dir)
end

input_file = joinpath(input_dir, "loss.bson")
data = BSON.load(input_file)[:loss]

ppo_loss = data["ppo"]
entropy_loss = data["entropy"]

fig, ax = plt.subplots()
ax.plot(ppo_loss)
ax.grid()
ax.set_title("Loss history of PPO objective")
ax.set_xlabel("PPO Iterations")
ax.set_ylabel("PPO Objective")
fig.tight_layout()
fig
output_file = joinpath(output_dir, "ppo_loss_history.png")
fig.savefig(output_file)

fig, ax = plt.subplots()
ax.plot(entropy_loss)
ax.grid()
ax.set_title("Loss history of entropy regularization")
ax.set_xlabel("PPO Iterations")
ax.set_ylabel("-ve Policy entropy")
fig.tight_layout()
fig
output_file = joinpath(output_dir, "entropy_loss_history.png")
fig.savefig(output_file)