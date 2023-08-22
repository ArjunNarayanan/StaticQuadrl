mutable struct FixedMeshEnv <: AbstractGameEnv
    mesh0
    desired_degree
    num_actions
    max_actions
    env
    current_score
    opt_score
    is_terminated
    reward
    cleanup
    local2global_half_edges
    function FixedMeshEnv(
        mesh, 
        desired_degree, 
        max_actions, 
        cleanup,
    )
        @assert max_actions > 0
        mesh0 = deepcopy(mesh)
        d0 = deepcopy(desired_degree)
        env = QM.GameEnv(mesh, desired_degree)
        current_score = global_score(env.vertex_score)
        opt_score = optimal_score(env.vertex_score)
        reward = 0
        num_actions = 0
        is_terminated = check_terminated(current_score, opt_score, num_actions, max_actions)
        local2global_half_edges = zeros(Int, NUM_EDGES_PER_TEMPLATE)

        new(
            mesh0, 
            d0, 
            num_actions, 
            max_actions, 
            env, 
            current_score, 
            opt_score, 
            is_terminated, 
            reward,
            cleanup,
            local2global_half_edges
        )
    end
end

function Base.show(io::IO, wrapper::FixedMeshEnv)
    println(io, "FixedMeshEnv")
    println(io, "\t$(wrapper.max_actions) max actions")
    show(io, wrapper.env)
end

function PPO.reset!(wrapper::FixedMeshEnv)
    mesh = deepcopy(wrapper.mesh0)
    d0 = deepcopy(wrapper.desired_degree)
    wrapper.env = QM.GameEnv(mesh, d0)
    wrapper.current_score = global_score(wrapper.env.vertex_score)
    wrapper.reward = 0
    wrapper.num_actions = 0
    wrapper.opt_score = optimal_score(wrapper.env.vertex_score)
    wrapper.is_terminated = check_terminated(wrapper.current_score, wrapper.opt_score,
        wrapper.num_actions, wrapper.max_actions)
    fill!(wrapper.local2global_half_edges, 0)
end
