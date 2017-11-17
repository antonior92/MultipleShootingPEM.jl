function simulate_space_state!(y, f::Function, g::Function, x0,
        time_span::Tuple{Int, Int}, args::Tuple=();
        buffer1=copy(x0), buffer2=copy(x0))
    # define initial state
    x = buffer1
    x_next = buffer2
    aux = buffer2
    # Initialize with x0
    x .= x0
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
end
