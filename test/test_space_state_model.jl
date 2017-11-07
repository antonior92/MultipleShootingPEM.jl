@testset "Test space state_simulate" begin
    # Declare functions
    function f(x_next, x, k)
        x_next[1] = x[2]
        x_next[2] = x[1]
    end
    function g(y, x, k)
        y[1] = x[1]
    end
    # Initialize vector
    time_span = (1, 5)
    x0 = [1, 0]
    y = Vector{typeof(x0)}(time_span[2] - time_span[1]+1)
    for i = 1:length(y)
        y[i] = [0]
    end
    buffer1 = copy(x0)
    buffer2 = copy(x0)
    # Simulate
    simulate_space_state!(f, g, x0, time_span, y;
        buffer1=buffer1, buffer2=buffer2)
    # Test
    @test y == [[1], [0], [1], [0], [1]]
end
