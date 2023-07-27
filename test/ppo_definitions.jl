function PPO.state(wrapper)
    return SQ.state(wrapper)
end

function PPO.reset!(wrapper)
    SQ.reset!(wrapper)
end

function PPO.number_of_actions_per_state(state)
    vs = state.half_edge_features
    @assert ndims(vs) == 3
    num_actions = size(vs, 2) * SQ.NUM_ACTIONS_PER_EDGE
    return num_actions
end

function PPO.batch_advantage(state, returns)
    return returns ./ state.optimum_return
end

function PPO.action_probabilities(policy, state)
    return SQ.action_probabilities(policy, state)
end

function PPO.step!(wrapper, index)
    SQ.step!(wrapper, index)
end