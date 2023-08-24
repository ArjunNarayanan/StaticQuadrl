mutable struct RandPolyEnv <: AbstractGameEnv
    poly_degree_list
    poly_degree
    quad_alg
    num_actions
    max_actions_factor
    max_actions::Any
    env::Any
    current_score
    opt_score
    is_terminated
    reward
    cleanup
    round_desired_degree
    local2global_half_edges
    function RandPolyEnv(
        poly_degree_list,
        max_actions_factor,
        quad_alg,
        cleanup,
        round_desired_degree
    )
        @assert max_actions_factor > 0
        @assert all(poly_degree_list .> 3)

        poly_degree = rand(poly_degree_list)
        max_actions = max_actions_factor * poly_degree
        mesh, d0 = initialize_random_mesh(poly_degree, quad_alg, round_desired_degree)
        env = QM.GameEnv(mesh, d0)
        current_score = global_score(env.vertex_score)
        opt_score = optimal_score(env.vertex_score)
        reward = 0.0f0
        num_actions = 0
        is_terminated = check_terminated(current_score, opt_score, num_actions, max_actions)
        local2global_half_edges = zeros(Int, NUM_EDGES_PER_TEMPLATE)

        new(
            poly_degree_list,
            poly_degree,
            quad_alg,
            num_actions,
            max_actions_factor,
            max_actions,
            env,
            current_score,
            opt_score,
            is_terminated,
            reward,
            cleanup,
            round_desired_degree,
            local2global_half_edges
        )
    end
end

function Base.show(io::IO, wrapper::RandPolyEnv)
    println(io, "RandPolyEnv")
    println(io, "\t$(wrapper.poly_degree) polygon degree")
    println(io, "\t$(wrapper.max_actions) max actions")
    show(io, wrapper.env)
end

function PPO.is_terminal(wrapper::RandPolyEnv)
    return wrapper.is_terminated
end

function PPO.reward(wrapper::RandPolyEnv)
    return wrapper.reward
end

function PPO.reset!(wrapper::RandPolyEnv)
    wrapper.poly_degree = rand(wrapper.poly_degree_list)
    wrapper.max_actions = wrapper.max_actions_factor * wrapper.poly_degree
    mesh, d0 = initialize_random_mesh(
        wrapper.poly_degree,
        wrapper.quad_alg,
        wrapper.round_desired_degree
    )
    wrapper.env = QM.GameEnv(mesh, d0)
    wrapper.current_score = global_score(wrapper.env.vertex_score)
    wrapper.reward = 0
    wrapper.num_actions = 0
    wrapper.opt_score = optimal_score(wrapper.env.vertex_score)
    wrapper.is_terminated = check_terminated(wrapper.current_score, wrapper.opt_score,
        wrapper.num_actions, wrapper.max_actions)
    fill!(wrapper.local2global_half_edges, 0)
    return
end

function _update_env_after_step!(wrapper)
    wrapper.current_score = global_score(wrapper.env.vertex_score)
    wrapper.num_actions += 1

    if !is_valid_mesh(wrapper.env.mesh)
        terminate_invalid_environment(wrapper)
    else
        wrapper.is_terminated = check_terminated(
            wrapper.current_score,
            wrapper.opt_score,
            wrapper.num_actions,
            wrapper.max_actions
        )
    end

    if wrapper.cleanup
        maxsteps = 2 * QM.number_of_quads(wrapper.env.mesh)
        QM.cleanup_env!(wrapper.env, maxsteps)
    end
end

function update_env_after_step!(wrapper, success)
    previous_score = wrapper.current_score
    wrapper.current_score = global_score(wrapper.env.vertex_score)
    wrapper.num_actions += 1

    wrapper.is_terminated = check_terminated(
        wrapper.current_score,
        wrapper.opt_score,
        wrapper.num_actions,
        wrapper.max_actions
    )

    if success
        wrapper.reward = previous_score - wrapper.current_score
    else
        wrapper.reward = NO_ACTION_REWARD
    end

    if !is_valid_mesh(wrapper.env.mesh)
        terminate_invalid_environment(wrapper)
    else
        if wrapper.cleanup
            try
                maxsteps = 2 * QM.number_of_quads(wrapper.env.mesh)
                QM.cleanup_env!(wrapper.env, maxsteps)
            catch e
                terminate_invalid_environment(wrapper)
            end
        end
    end

end
