function action_probabilities(policy, state)
    @assert policy.num_output == NUM_ACTIONS_PER_EDGE

    features = state.half_edge_features
    logits = vec(policy(features))
    p = softmax(logits)

    return p
end