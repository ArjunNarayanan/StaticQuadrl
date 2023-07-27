struct SimplePolicy
    model
    in_channels
    hidden_channels
    num_hidden_layers
    num_output
end

function SimplePolicy(
    in_channels, 
    hidden_channels, 
    num_hidden_layers, 
    num_output
)
    model = []
    push!(model, Dense(in_channels, hidden_channels, leakyrelu))
    for i in 1:num_hidden_layers-1
        push!(model, Dense(hidden_channels, hidden_channels, leakyrelu))
    end
    push!(model, Dense(hidden_channels, num_output))
    model = Chain(model...)

    SimplePolicy(
        model,
        in_channels, 
        hidden_channels, 
        num_hidden_layers, 
        num_output
    )
end

Flux.@functor SimplePolicy

function Base.show(io::IO, p::SimplePolicy)
    s = "Policy\n\t$(p.hidden_channels) channels\n\t$(p.num_hidden_layers) layers"
    println(io, s)
end

function (p::SimplePolicy)(state)
    return p.model(state)
end
