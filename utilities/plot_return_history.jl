using BSON
include("../src/StaticQuadrl.jl")
using PyPlot
using Printf

model_dir = ARGS[1]
input_dir = model_dir
output_dir = joinpath(input_dir, "figures")

if !isdir(output_dir)
    mkpath(output_dir)
end

saved_data = joinpath(input_dir, "evaluator.bson")
data = BSON.load(saved_data)[:data]
evaluator = data["evaluator"]

mean_returns = evaluator.mean_returns
max_returns = maximum(mean_returns)
print("Max performance : ", @sprintf "%1.3f\n" max_returns)
dev = evaluator.std_returns

lower_bound = mean_returns - dev
upper_bound = mean_returns + dev

fig, ax = subplots()
ax.plot(mean_returns)
ax.fill_between(1:length(mean_returns),lower_bound, upper_bound, alpha = 0.4)
ax.grid()
ax.set_ylim([-1,1])
ax.set_xlabel("PPO Iterations")
ax.set_ylabel("Mean Returns")
fig.tight_layout()
fig

output_file = joinpath(output_dir, "returns.pdf")
fig.savefig(output_file)