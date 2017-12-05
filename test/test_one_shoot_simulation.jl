@testset "Test on logistic map" begin
    # Define logistic map
    function logistic_f(x_next, x, k, θ)
        x_next .=  θ*x - θ*x.^2
        return
    end
    # g(x) = x
    function logistic_g(y, x, k, θ)
        y .= x
        return
    end
    # Define initial conditions
    x0 = [0.5]
    # Initialize buffer
    y = Vector{Vector{Float64}}(100)
    for i = 1:length(y)
        y[i] = zeros(1)
    end
    y2 = Vector{Vector{Float64}}(100)
    for i = 1:length(y)
        y2[i] = zeros(1)
    end
    # Simulate
    x_final = ms.simulate_space_state!(y, logistic_f, logistic_g,
                                       [0.5], (1, 100), (3.78,))
    x_final2 = ms.simulate_space_state!(y2, logistic_f, logistic_g,
                                       [0.5], (1, 100), (3.62,))

    # Define initial values
    time_span = (1, 100)
    Ny = 1
    N = 100
    x0 = [0.5]
    θ = [3.78, 3.78, 1.00]
    θ2 = [3.62, 3.62, 1.00]
    # Define polynomial model
    # f(x) = θ1*x - θ2*x^2
    function f(y, dx, dθ, x, k, θ)
        y .= θ[1]*x - θ[2]*x.^2
        dx .= θ[1] - 2*θ[2]*x
        dθ .=  [ x  -x.^2  0]
        return
    end
    # g(x) = x
    function g(y, dx, dθ, x, k, θ)
        y .= θ[3]*x
        dx .= θ[3]
        dθ .=  [ 0 0 x ]
        return
    end

    oss = ms.OneShootSimulation(f, g, x0, time_span, Ny, θ)

    @testset "Test function evaluation" begin
        @test vcat(y...) - vcat(oss.ys) ≈ zeros(100)
        @test oss.x ≈ x_final

        ms.new_simulation(oss, x0, θ2)

        @test vcat(y2...) - vcat(oss.ys) ≈ zeros(100)
        @test oss.x ≈ x_final2
    end

    @testset "Test jacobian" begin
        function wrapper_simulation(ϕ)
            θ1 = ϕ[1:3]
            x01 = [ϕ[4]]
            ms.new_simulation(oss, x01, θ1)
            return vcat(oss.ys...)
        end

        function wrapper_jacobian(ϕ)
           θ1 = ϕ[1:3]
           x01 = [ϕ[4]]
           ms.new_simulation(oss, x01, θ1)
           return hcat(vcat(oss.dydθ...), vcat(oss.dydx0...))
        end

        function finite_difference_jacobian(ϕ)
            return jacobian(wrapper_simulation, ϕ, :central)
        end

       # OBS: finite difference approximation
       # fails when trying parameters in the
       # chaotic region
       for ϕ in ([3.21, 3.2, 1.00, 0.5],
                 [3.21, 1.2, 1.54, 0.5],
                 [2.56, 3.27, 1.54, 0.33])
            @test wrapper_jacobian(ϕ) ≈ finite_difference_jacobian(ϕ)
       end
    end

    @testset "Test cost function" begin
        ms.new_simulation(oss, x0, θ)
        @test ms.cost_function(y, oss) ≈ 0
        ms.new_simulation(oss, x0, θ2)
        @test ms.cost_function(y2, oss) ≈ 0
    end

    @testset "Test gradient" begin
        function wrapper_cost(ϕ)
            θ1 = ϕ[1:3]
            x01 = [ϕ[4]]
            ms.new_simulation(oss, x01, θ1)
            return ms.cost_function(y, oss)
        end

        function wrapper_gradient(ϕ)
            θ1 = ϕ[1:3]
            x01 = [ϕ[4]]
            ms.new_simulation(oss, x01, θ1)
            gradθ = zeros(3)
            ms.gradient!(gradθ, y, oss, "θ")
            gradx0 = zeros(1)
            ms.gradient!(gradx0, y, oss, "x0")
            return vcat(gradθ, gradx0)
        end

        function finite_difference_gradient(ϕ)
            return Calculus.gradient(wrapper_cost, ϕ)
        end

        # OBS: finite difference approximation
        # fails when trying parameters in the
        # chaotic region
        for ϕ in ([3.21, 3.2, 1.00, 0.5],
                 [3.21, 1.2, 1.54, 0.5],
                 [2.56, 3.27, 1.54, 0.33])
            @test wrapper_gradient(ϕ) ≈ finite_difference_gradient(ϕ)
        end
    end
end

@testset "Test on linear problem" begin
    # From example 2.11, "Linear System Theory and design" Chen
    # With not very realistic values
    # Define linear function
    function linear_f(x_next, x, k)
        R = 20
        C1 = 10
        C2 = 10
        L = 1
        A = [-1/(R*C1) 0   -1/C1;
             0         0    1/C2;
             1/L       -1/L    0]
        x_next .=  A*x
        return
    end
    # g(x) = x
    function linear_g(y, x, k)
      y .= dot([1, -1, 0], x)
      return
    end
    # Define initial conditions
    x0 = [5, 0.6, 0.5]
    # Initialize buffer
    y = Vector{Vector{Float64}}(10)
    for i = 1:length(y)
       y[i] = zeros(1)
    end
    # Simulate
    x_final = ms.simulate_space_state!(y, linear_f, linear_g,
                                       x0, (1, 10))

    # Define initial values
    time_span = (1, 10)
    Ny = 1
    N = 10
    x0 = [5, 0.6, 0.5]
    θ = [-0.005, 0.1, 0.1, 1]
    # Define polynomial model
    # f(x) = θ1*x - θ2*x^2
    function f(y, dx, dθ, x, k, θ)
        A = [θ[1]  0 -θ[2];
             0     0  θ[3];
             θ[4] -θ[4]  0]
         y .= A*x
         dx .= A
         dθ .= [x[1] -x[3]    0      0;
                0      0      x[3]   0;
                0      0       0    x[1]-x[2]]
         return
    end
    # g(x) = x
    function g(y, dx, dθ, x, k, θ)
      y .= dot([1, -1, 0], x)
      dx .= [1 -1 0]
      dθ .=  0
      return
    end

    oss = ms.OneShootSimulation(f, g, x0, time_span, Ny, θ)

    @testset "Test function evaluation" begin
       @test vcat(y...) - vcat(oss.ys) ≈ zeros(10)
       @test oss.x ≈ x_final
    end


    @testset "Test jacobian" begin
        function wrapper_simulation(ϕ)
           θ1 = ϕ[1:4]
           x01 = ϕ[5:7]
           ms.new_simulation(oss, x01, θ1)
           return vcat(oss.ys...)
        end

        function wrapper_jacobian(ϕ)
           θ1 = ϕ[1:4]
           x01 = ϕ[5:7]
           ms.new_simulation(oss, x01, θ1)
           return hcat(vcat(oss.dydθ...), vcat(oss.dydx0...))
        end

        function finite_difference_jacobian(ϕ)
            return jacobian(wrapper_simulation, ϕ, :central)
        end

        for ϕ in ([-0.005, 0.1, 0.1, 1, 5, 0.6, 0.5],
                  [-0.03, 0.11, 0.1, 2, 5, 0, 0.5],
                  [-0.5, 1.11, 1, 2, 2, 10, 0.5])
            @test wrapper_jacobian(ϕ) ≈ finite_difference_jacobian(ϕ)
        end
    end

    @testset "Test cost function" begin
       ms.new_simulation(oss, x0, θ)
       @test ms.cost_function(y, oss) ≈ 0
    end

    function wrapper_cost(ϕ)
       θ1 = ϕ[1:4]
       x01 = ϕ[5:7]
       ms.new_simulation(oss, x01, θ1)
       return ms.cost_function(y, oss)
    end

    @testset "Test gradient" begin
       function wrapper_gradient(ϕ)
          θ1 = ϕ[1:4]
          x01 = ϕ[5:7]
          ms.new_simulation(oss, x01, θ1)
          gradθ = zeros(4)
          ms.gradient!(gradθ, y, oss, "θ")
          gradx0 = zeros(3)
          ms.gradient!(gradx0, y, oss, "x0")
          return vcat(gradθ, gradx0)
       end
       function finite_difference_gradient(ϕ)
          return Calculus.gradient(wrapper_cost, ϕ)
       end
       for ϕ in ([-0.005, 0.1, 0.1, 1, 5, 0.6, 0.5],
                 [-0.03, 0.11, 0.1, 2, 5, 0, 0.5],
                 [-0.5, 1.11, 1, 2, 2, 10, 0.5])
          @test wrapper_gradient(ϕ) ≈ finite_difference_gradient(ϕ) atol=1e-10
       end
    end
end
