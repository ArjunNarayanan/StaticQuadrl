function local_half_edge_index_and_action_type(index)
    local_half_edge = div(index - 1, NUM_EDGES_PER_ELEMENT) + 1
    action_type = rem(index - 1, NUM_ACTIONS_PER_EDGE) + 1
    return local_half_edge, action_type
end

function global_quad_and_edge_index(global_half_edge_index)
    quad = div(global_half_edge_index - 1, NUM_EDGES_PER_ELEMENT) + 1
    edge = rem(global_half_edge_index - 1, NUM_EDGES_PER_ELEMENT) + 1

    return quad, edge
end

function step!(wrapper, local_action_index)
    @assert !wrapper.is_terminated "Attempting to step in terminated environment with action $action_index"
    local_half_edge, action_type = local_half_edge_index_and_action_type(local_action_index)
    global_half_edge_index = wrapper.local2global_half_edges[local_half_edge]
    quad, edge = global_quad_and_edge_index(global_half_edge_index)

    step_wrapper!(wrapper, quad, edge, action_type)
end

function terminate_invalid_environment(wrapper)
    opt_return = wrapper.current_score - wrapper.opt_score
    # set the reward such that the normalized reward is -1
    wrapper.reward = -1.0 * opt_return
    wrapper.is_terminated = true
end

function is_valid_mesh(mesh)
    return QM.all_active_vertices(mesh) && 
    QM.no_quad_self_reference(mesh) &&
    QM.all_active_quad_or_boundary(mesh)
end

function step_wrapper!(wrapper, quad, edge, type)
    env = wrapper.env
    previous_score = wrapper.current_score
    success = false

    # @assert QM.is_active_quad(env.mesh, quad) "Attempting to act on inactive quad $quad with action ($quad, $edge, $type)"
    @assert type in 1:NUM_ACTIONS_PER_EDGE "Expected action type in {1,...,$NUM_ACTIONS_PER_EDGE} got type = $type"
    @assert edge in (1, 2, 3, 4) "Expected edge in {1,2,3,4} got edge = $edge"
    @assert isapprox(wrapper.opt_score, optimal_score(wrapper.env.vertex_score))


    if !is_valid_mesh(env.mesh)
        terminate_invalid_environment(wrapper)
        return
    elseif !QM.is_active_quad(env.mesh, quad)
        success = false
    elseif type == 1
        success = QM.step_left_flip!(env, quad, edge)
    elseif type == 2
        success = QM.step_right_flip!(env, quad, edge)
    elseif type == 3
        success = QM.step_split!(env, quad, edge)
    elseif type == 4
        success = QM.step_collapse!(env, quad, edge)
    elseif type == 5
        maxsplits = 2*QM.number_of_quads(env.mesh)
        success = QM.step_global_split_without_loops!(env, quad, edge, maxsplits)
    else
        error("Unexpected action type $type")
    end

    update_env_after_step!(wrapper)
    
    if success
        wrapper.reward = previous_score - wrapper.current_score
    else
        wrapper.reward = NO_ACTION_REWARD
    end
    
end
