using StaticQuadrl

include("plot.jl")

SQ = StaticQuadrl

wrapper = SQ.RandPolyEnv(
    [20],
    3,
    "catmull-clark",
    true,
    false,
)


plot_wrapper(
    wrapper,
    number_vertices=true,
    internal_order=true
)
