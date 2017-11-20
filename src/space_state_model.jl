# Simulate space state model
# x0 (and x0_2) will be overwritten during execution
function simulate_space_state!(y::Vector, f::Function, g::Function, x0,
        time_span::Tuple{Int, Int}, args::Tuple=(); x0_2=deepcopy(x0))

    # Define buffers
    if mod(time_span[2]-time_span[1]+1, 2) == 0
        x = x0
        x_next = x0_2
    else
        x = x0_2
        x_next = x0
    end
    aux = x0_2
    # Iterate using recursive equation
    i = 1
    for k = time_span[1]:time_span[2]
        # Evaluate function
        f(x_next, x, k, args...)
        g(y[i], x, k, args...)
        # Swap buffers
        aux = x
        x = x_next
        x_next = aux
        # Next iteration
        i+=1
    end
    return x
end
