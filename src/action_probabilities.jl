function PPO.action_probabilities(
    policy::SimplePolicy, 
    state::StateData
)

    features = state.half_edge_features
    logits = vec(policy(features))
    p = softmax(logits)

    return p
end

function PPO.batch_action_probabilities(
    policy::SimplePolicy, 
    state::StateData
)
    features = state.half_edge_features
    nf, nq, nb = size(features)
    logits = reshape(policy(features), :, nb)
    probs = softmax(logits, dims=1)
    return probs
end