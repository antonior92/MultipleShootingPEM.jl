@testset "Test space state simulation" begin
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
    x = simulate_space_state!(y, f, g, x0, time_span;
            buffer1=buffer1, buffer2=buffer2)
    # Test
    @test y == [[1], [0], [1], [0], [1]]
    @test x == [0, 1]
end


@testset "Test space state simulation extra args" begin
    # Declare functions
    function f(x_next, x, k, a, b, c)
        x_next[1] = a[1]*b[1]*c[2]*x[2]
        x_next[2] = a[1]*b[1]*c[2]*x[1]
    end
    function g(y, x, k, a, b, c)
        y[1] = a[2]*b[2]*c[1]*x[1]
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
    a = Float64[1, 1]
    b = Float64[1, 1]
    c = Float64[1, 1]
    args = (a, b, c)
    x = simulate_space_state!(y, f, g, x0, time_span, args;
        buffer1=buffer1, buffer2=buffer2)
    # Test
    @test y == [[1], [0], [1], [0], [1]]
    @test x == [0, 1]
end
