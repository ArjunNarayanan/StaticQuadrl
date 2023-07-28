#####################################################################################################################
# TEMPLATE GENERATION

function make_level4_template(wrapper)
    template = QM.make_level4_template(wrapper.env.mesh)
    return template
end
#####################################################################################################################

#####################################################################################################################
# GENERATING AND MANIPULATING ENVIRONMENT STATE
struct StateData
    half_edge_features
    global_half_edge_indices
    optimum_return
end

function Base.show(io::IO, s::StateData)
    println(io, "StateData")
end

function Flux.gpu(s::StateData)
    return StateData(
        gpu(s.half_edge_features),
        s.global_half_edge_indices,
        s.optimum_return
    )
end

function Flux.cpu(s::StateData)
    return StateData(
        cpu(s.half_edge_features), 
        s.global_half_edge_indices, 
        s.optimum_return,
    )
end

function PPO.state(wrapper::RandPolyEnv)
    half_edge_template = QM.level4_half_edge_template(
        wrapper.env.mesh
    )
    connectivity = vec(wrapper.env.mesh.connectivity)
    push!(connectivity, 0)

    missing_index = length(connectivity)
    half_edge_template[half_edge_template.==0] .= missing_index

    vertex_template = connectivity[half_edge_template]
    vertex_score = Float32.(wrapper.env.vertex_score)
    push!(vertex_score, 0.0f0)
    vertex_degree = Float32.(wrapper.env.mesh.degree)
    push!(vertex_degree, 0.0f0)
    @assert length(vertex_score) == length(vertex_degree)

    missing_index = length(vertex_score)
    vertex_template[vertex_template.==0] .= missing_index

    vs = vertex_score[vertex_template]
    vd = vertex_degree[vertex_template]

    template_scores = vec(
        sum(abs.(vs), dims=1)
    )
    half_edge_index = rand(
        findall(template_scores .== maximum(template_scores))
    )

    local_half_edges = half_edge_template[:, half_edge_index]
    wrapper.local2global_half_edges .= local_half_edges

    vs = QM.zero_pad_matrix_cols(vs, 1)
    vd = QM.zero_pad_matrix_cols(vd, 1)
    vs = vs[:,local_half_edges]
    vd = vd[:,local_half_edges]

    half_edge_features = vcat(vs, vd)
    opt_return = wrapper.current_score - wrapper.opt_score

    return StateData(
        half_edge_features,
        local_half_edges,
        opt_return
    )
end
#####################################################################################################################


#####################################################################################################################
# BATCHING STATE
function PPO.batch_state(state_data_vector::Vector{StateData})
    features = [s.half_edge_features for s in state_data_vector]
    half_edge_indices = [s.global_half_edge_indices for s in state_data_vector]
    opt_return = [s.optimum_return for s in state_data_vector]
    batch_features = cat(features..., dims=3)
    batch_half_edge_indices = cat(half_edge_indices..., dims=2)
    return StateData(batch_features, batch_half_edge_indices, opt_return)
end

function PPO.number_of_actions_per_state(state::StateData)
    vs = state.half_edge_features
    @assert ndims(vs) == 3
    num_actions = size(vs, 2) * NUM_ACTIONS_PER_EDGE
    return num_actions
end

function PPO.batch_advantage(state::StateData, returns)
    return returns ./ state.optimum_return
end
#####################################################################################################################