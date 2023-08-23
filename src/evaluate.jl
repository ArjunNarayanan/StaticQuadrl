#####################################################################################################################
# EVALUATING PERFORMANCE
mutable struct SaveBestModel
    root_dir
    file_path
    num_trajectories
    best_return
    mean_returns
    std_returns
    action_counts
    function SaveBestModel(root_dir, num_trajectories, filename = "best_model.bson")
        if !isdir(root_dir)
            mkpath(root_dir)
        end

        file_path = joinpath(root_dir, filename)
        mean_returns = []
        std_returns = []
        action_counts = []
        new(root_dir, file_path, num_trajectories, -Inf, mean_returns, std_returns, action_counts)
    end
end

function save_model(s::SaveBestModel, policy)
    cpu_policy = cpu(policy)
    data = Dict("policy" => cpu_policy)
    println("SAVING MODEL AT : " * s.file_path * "\n\n")
    BSON.@save s.file_path data
end

function save_model_and_optimizer(s::SaveBestModel, policy, optimizer)
    cpu_policy = cpu(policy)
    data = Dict("policy" => cpu_policy, "optimizer" => optimizer)
    println("SAVING MODEL AT : " * s.file_path * "\n\n")
    BSON.@save s.file_path data
end

function save_evaluator(s::SaveBestModel)
    data = Dict("evaluator" => s)
    filepath = joinpath(s.root_dir, "evaluator.bson")
    println("SAVING EVALUATOR AT : ", filepath, "\n")
    BSON.@save filepath data
end

function (s::SaveBestModel)(policy, wrapper, optimizer)
    ret, dev, action_counts = average_normalized_returns_and_action_stats(policy, wrapper, s.num_trajectories)
    if ret > s.best_return
        s.best_return = ret
        @printf "\nNEW BEST RETURN : %1.4f\n" ret
        save_model_and_optimizer(s, policy, optimizer)
    end

    @printf "RET = %1.4f\tDEV = %1.4f\n" ret dev
    println("ACTION COUNTS: ", action_counts, "\n")
    push!(s.mean_returns, ret)
    push!(s.std_returns, dev)
    push!(s.action_counts, action_counts)

    save_evaluator(s)
end

function PPO.save_loss(s::SaveBestModel, loss)
    outfile = joinpath(s.root_dir, "loss.bson")
    BSON.@save outfile loss
end

function single_trajectory_return_and_action_stats(policy, env)
    ret = 0
    action_type_list = Int[]
    done = PPO.is_terminal(env)

    while !done
        state = PPO.state(env) |> gpu
        probs = PPO.action_probabilities(policy, state) |> cpu
        action = rand(Categorical(probs))
        @assert probs[action] > 0.0
        _, action_type = local_half_edge_index_and_action_type(action)

        PPO.step!(env, action)

        done = PPO.is_terminal(env)
        ret += PPO.reward(env)
        push!(action_type_list, action_type)
    end
    return ret, action_type_list
end

function single_trajectory_normalized_return_and_action_stats(policy, wrapper)
    maxreturn = wrapper.current_score - wrapper.opt_score
    if maxreturn == 0
        return 1.0, Int[]
    else
        ret, stats = single_trajectory_return_and_action_stats(policy, wrapper)
        return ret / maxreturn, stats
    end
end

function average_normalized_returns_and_action_stats(policy, wrapper, num_trajectories)
    ret = zeros(num_trajectories)
    actions = Int[]
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        r, s = single_trajectory_normalized_return_and_action_stats(policy, wrapper)
        ret[idx] = r
        append!(actions, s)
    end
    mean = Flux.mean(ret)
    std = Flux.std(ret)
    stats = [count(actions .== i) for i = 1:NUM_ACTIONS_PER_EDGE]

    return mean, std, stats
end

function single_trajectory_score_history(policy, wrapper)
    done = PPO.is_terminal(wrapper)
    
    T = typeof(wrapper.current_score)
    score_history = T[]

    while !done
        state = PPO.state(wrapper) |> gpu
        probs = PPO.action_probabilities(policy, state) |> cpu
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)

        push!(score_history, wrapper.current_score)
        done = PPO.is_terminal(wrapper)
    end
    return score_history
end

function single_trajectory_normalized_return_history(policy, wrapper)
    initial_score = wrapper.current_score
    opt_score = wrapper.opt_score

    current_scores = single_trajectory_score_history(policy, wrapper)
    normalized_ret = (initial_score .- current_scores) ./ (initial_score - opt_score)
    return normalized_ret
end

function best_single_trajectory_return(policy, wrapper)
    done = PPO.is_terminal(wrapper)

    initial_score = wrapper.current_score
    minscore = wrapper.current_score

    while !done
        state = PPO.state(wrapper) |> gpu
        probs = PPO.action_probabilities(policy, state) |> cpu
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)

        minscore = min(minscore, wrapper.current_score)
        done = PPO.is_terminal(wrapper)
    end
    return initial_score - minscore
end

function best_normalized_single_trajectory_return(policy, wrapper)
    max_return = wrapper.current_score - wrapper.opt_score
    if max_return == 0
        return 1.0
    else
        ret = best_single_trajectory_return(policy, wrapper)
        return ret/max_return
    end
end

function average_normalized_best_returns(policy, wrapper, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = best_normalized_single_trajectory_return(policy, wrapper)
    end
    return Flux.mean(ret), Flux.std(ret)
end
