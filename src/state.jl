#####################################################################################################################
# TEMPLATE GENERATION
function pad_matrix_cols(mat, num_new_cols, value)
    nr, _ = size(mat)
    return [mat fill(value, (nr, num_new_cols))]
end

function zero_pad_matrix_cols(m, num_new_cols)
    T = eltype(m)
    return pad_matrix_cols(m, num_new_cols, zero(T))
end

function cycle_edges(x)
    nf, na = size(x)
    x = reshape(x, nf, 4, :)

    x1 = reshape(x, 4nf, 1, :)
    x2 = reshape(x[:, [2, 3, 4, 1], :], 4nf, 1, :)
    x3 = reshape(x[:, [3, 4, 1, 2], :], 4nf, 1, :)
    x4 = reshape(x[:, [4, 1, 2, 3], :], 4nf, 1, :)

    x = cat(x1, x2, x3, x4, dims=2)
    x = reshape(x, 4nf, :)

    return x
end

function make_level4_template(pairs, x)
    cx = cycle_edges(x)

    pcx = zero_pad_matrix_cols(cx, 1)[:, pairs][3:end, :]
    cpcx = cycle_edges(pcx)

    pcpcx = zero_pad_matrix_cols(cpcx, 1)[:, pairs][3:end, :]
    cpcpcx = cycle_edges(pcpcx)

    pcpcpcx = zero_pad_matrix_cols(cpcpcx, 1)[:, pairs][7:end, :]
    cpcpcpcx = cycle_edges(pcpcpcx)

    template = vcat(cx, cpcx, cpcpcx, cpcpcpcx)

    return template
end


function level4_active_template(wrapper)
    @assert !wrapper_requires_reindex(wrapper)
    num_quads = QM.number_of_quads(wrapper.env.mesh)
    active_half_edges = 1:4*num_quads
    template = QM.make_level4_template(wrapper.env.mesh)
    active_template = template[:, active_half_edges]
    return active_template
end
#####################################################################################################################

#####################################################################################################################
# GENERATING AND MANIPULATING ENVIRONMENT STATE
struct StateData
    env
    half_edge_template
    vertex_score
    vertex_degree
    global_half_edge_indices
end

function Base.show(io::IO, s::StateData)
    println(io, "StateData")
end

# function Flux.gpu(s::StateData)
#     return StateData(
#         gpu(s.vertex_score), 
#         gpu(s.action_mask), 
#         s.optimum_return,
#         s.env
#     )
# end
#####################################################################################################################