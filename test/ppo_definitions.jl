function PPO.state(wrapper)
    return SQ.get_state(wrapper)
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

function PPO.batch_action_probabilities(policy, state)
    return SQ.batch_action_probabilities(policy, state)
end

function PPO.step!(wrapper, index)
    SQ.step!(wrapper, index)
end

function PPO.is_terminal(wrapper)
    return SQ.is_terminal(wrapper)
end

function PPO.reward(wrapper)
    return wrapper.reward
end

function PPO.batch_state(state_data_vector)
    return SQ.batch_state(state_data_vector)
end

function PPO.save_loss(evaluator, loss)
    SQ.save_loss(evaluator, loss)
end