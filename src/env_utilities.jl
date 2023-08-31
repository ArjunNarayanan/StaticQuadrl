abstract type AbstractGameEnv end

function _vanilla_global_score(vertex_score)
    return sum(abs.(vertex_score))
end

function _distance_weighted_global_score(vertex_score, distances)
    score = sum(abs.(vertex_score) .* distances)
    return score
end

function global_score(vertex_score)
    return _vanilla_global_score(vertex_score)
end

function optimal_score(vertex_score)
    return abs(sum(vertex_score))
end

function check_terminated(current_score, opt_score, num_actions, max_actions)
    return (num_actions >= max_actions) || (current_score <= opt_score)
end

function generate_mesh_from_boundary(
    boundary_pts, 
    quad_alg, 
    round_desired_degree;
    hmax = Inf,
    allow_vertex_insert = false
)
    num_points = size(boundary_pts, 2)

    angles = QM.polygon_interior_angles(boundary_pts)

    bdry_d0 = round_desired_degree ? QM.rounded_desired_degree.(angles) : QM.continuous_desired_degree.(angles)

    mesh = RQ.quad_mesh(boundary_pts, algorithm=quad_alg, hmax = hmax, allow_vertex_insert = allow_vertex_insert)
    num_vertices = size(mesh.p, 2)
    is_geometric_vertex = falses(num_vertices)
    is_geometric_vertex[1:num_points] .= true

    mesh = QM.QuadMesh(mesh.p, mesh.t, is_geometric_vertex = is_geometric_vertex)

    mask = .![trues(num_points); falses(mesh.num_vertices - num_points)]
    mask = mask .& mesh.vertex_on_boundary[mesh.active_vertex]

    d0 = [bdry_d0; fill(4, mesh.num_vertices - num_points)]
    d0[mask] .= 3

    return mesh, d0
end

function initialize_random_mesh(poly_degree, quad_alg, round_desired_degree)
    boundary_pts = RQ.random_polygon(poly_degree)
    angles = QM.polygon_interior_angles(boundary_pts)

    bdry_d0 = round_desired_degree ? QM.rounded_desired_degree.(angles) : QM.continuous_desired_degree.(angles)

    mesh = RQ.quad_mesh(boundary_pts, algorithm=quad_alg)
    num_vertices = size(mesh.p, 2)
    is_geometric_vertex = falses(num_vertices)
    is_geometric_vertex[1:poly_degree] .= true

    mesh = QM.QuadMesh(mesh.p, mesh.t, is_geometric_vertex = is_geometric_vertex)

    mask = .![trues(poly_degree); falses(mesh.num_vertices - poly_degree)]
    mask = mask .& mesh.vertex_on_boundary[mesh.active_vertex]

    d0 = [bdry_d0; fill(4, mesh.num_vertices - poly_degree)]
    d0[mask] .= 3

    return mesh, d0
end

function PPO.is_terminal(wrapper::T) where {T<:AbstractGameEnv}
    return wrapper.is_terminated
end

function PPO.reward(wrapper::T) where {T<:AbstractGameEnv}
    return wrapper.reward
end