using PlotQuadMesh
using QuadMeshGame
using PyPlot
using Printf
using Distributions: Categorical

QM = QuadMeshGame
PQ = PlotQuadMesh
#####################################################################################################################
# PLOTTING STUFF
function plot_env_score!(ax, score; coords = (0.8, 0.8), fontsize = 50)
    tpars = Dict(
        :color => "black",
        :horizontalalignment => "center",
        :verticalalignment => "center",
        :fontsize => fontsize,
        :fontweight => "bold",
    )

    ax.text(coords[1], coords[2], score; tpars...)
end

function plot_env(
    env, 
    score, 
    number_elements = false, 
    internal_order = false,
    mark_geometric_vertices=false,
)
    env = deepcopy(env)

    QM.reindex_game_env!(env)
    mesh = env.mesh
    vs = QM.active_vertex_score(env)

    fig, ax = PQ.plot_mesh(
        QM.active_vertex_coordinates(mesh),
        QM.active_quad_connectivity(mesh),
        vertex_score=vs,
        vertex_size = 30,
        number_elements = number_elements,
        internal_order = internal_order,
    )
    
    plot_env_score!(ax, score)

    return fig, ax
end

function plot_wrapper(
    wrapper; 
    filename = "", 
    xlim=nothing,
    ylim=nothing,
    smooth_iterations = 5, 
    number_elements = false, 
    mark_geometric_vertices = false,
    plot_score = true,
    boundary_smooth_iterations = 2,
    )

    smooth_wrapper!(wrapper, smooth_iterations)
    smooth_boundary!(wrapper, boundary_smooth_iterations)

    format_score(s) = isinteger(s) ? @sprintf("%1d", s) : @sprintf("%1.1f", s)
    cs = format_score(wrapper.current_score)
    os = format_score(wrapper.opt_score)
    text = plot_score ? cs * " / " * os : ""


    internal_order = number_elements
    element_numbers = number_elements ? findall(wrapper.env.mesh.active_quad) : false

    fig, ax = plot_env(
        wrapper.env, 
        text, 
        element_numbers, 
        internal_order, 
        mark_geometric_vertices
    )

    if isnothing(xlim)
        ax.set_xlim(-1, 1)
    else
        ax.set_xlim(xlim...)
    end

    if isnothing(ylim)
        ax.set_ylim(-1, 1)
    else
        ax.set_ylim(ylim...)
    end

    if length(filename) > 0
        fig.tight_layout()
        fig.savefig(filename)
    end

    return fig
end

function smooth_wrapper!(wrapper, num_iterations = 1)
    for iteration in 1:num_iterations
        QM.averagesmoothing!(wrapper.env.mesh)
    end
end

function plot_trajectory(
    policy, 
    wrapper, 
    root_directory;
    xlim=nothing,
    ylim=nothing,
    plot_score=true,
    extension = ".png"
)

    if !isdir(root_directory)
        mkpath(root_directory)
    end

    fig_name = "figure-" * lpad(0, 3, "0") * extension
    filename = joinpath(root_directory, fig_name)
    plot_wrapper(
        wrapper, 
        filename=filename,
        xlim=xlim,
        ylim=ylim,
        plot_score=plot_score
    )

    fig_index = 1
    done = PPO.is_terminal(wrapper)
    while !done 
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)
        
        fig_name = "figure-" * lpad(fig_index, 3, "0") * extension
        filename = joinpath(root_directory, fig_name)
        plot_wrapper(wrapper, filename=filename, xlim=xlim, ylim=ylim, plot_score=plot_score)
        fig_index += 1

        done = PPO.is_terminal(wrapper)
    end
end

function plot_returns(ret, lower_fill, upper_fill)
    fig, ax = subplots()
    ax.plot(ret)
    ax.fill_between(1:length(ret), lower_fill, upper_fill, alpha = 0.2, facecolor = "blue")
    ax.grid()
    ax.set_xlabel("PPO Iterations")
    ax.set_ylabel("Normalized returns")
    return fig, ax
end

function plot_normalized_returns(ret, dev)
    lower = ret - dev
    upper = ret + dev
    upper[upper .> 1.0] .= 1.0
    return plot_returns(ret, lower, upper)
end

#####################################################################################################################



#####################################################################################################################
# Smooth non geometric boundary vertices
function smooth_boundary!(wrapper, num_iter)
    for iter in 1:num_iter
        smooth_non_geometric_boundary_vertices!(wrapper.env.mesh)
    end
end

function smooth_non_geometric_boundary_vertices!(mesh)
    vertices = zeros(size(mesh.vertices))
    smooth_vertices = (.!mesh.is_geometric_vertex) .& (mesh.vertex_on_boundary)
    for quad in 1:QM.quad_buffer(mesh)
        if QM.is_active_quad(mesh, quad)
            for half_edge in 1:4
                source = QM.vertex(mesh, quad, half_edge)
                target = QM.vertex(mesh, quad, QM.next(half_edge))
                
                if smooth_vertices[source] && !QM.has_neighbor(mesh, quad, half_edge)
                    vertices[:, source] += mesh.vertices[:, target]
                end

                if smooth_vertices[target] && !QM.has_neighbor(mesh, quad, half_edge)
                    vertices[:, target] += mesh.vertices[:, source]
                end

            end
        end
    end
    # return vertices
    vertices = vertices/2
    mask = .!smooth_vertices
    vertices[:,mask] .= mesh.vertices[:,mask]
    mesh.vertices .= vertices
end

#####################################################################################################################