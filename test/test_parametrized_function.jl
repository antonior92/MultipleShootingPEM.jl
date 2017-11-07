@testset "Test sensitivity equation" begin
    function test_fun(y_extended, x, θ)
        y, dθ, dx = y_extended
        # y
        y[1] = θ[1]*x[1] + θ[3]*x[2] + θ[5]*x[3]
        y[2] = θ[2]*x[1] + θ[4]*x[2] + θ[6]*x[3]
        # dx
        dx[1, 1] = θ[1]
        dx[2, 1] = θ[2]
        dx[1, 2] = θ[3]
        dx[2, 2] = θ[4]
        dx[1, 3] = θ[5]
        dx[2, 3] = θ[6]
        # dθ
        dθ[1, 1] = x[1]
        dθ[1, 2] = 0.0
        dθ[1, 3] = x[2]
        dθ[1, 4] = 0.0
        dθ[1, 5] = x[3]
        dθ[1, 6] = 0.0
        dθ[2, 1] = 0.0
        dθ[2, 2] = x[1]
        dθ[2, 3] = 0.0
        dθ[2, 4] = x[2]
        dθ[2, 5] = 0.0
        dθ[2, 6] = x[3]
        return
    end

    jacobian_buffer = zeros(2, 3)

    θ = Float64[1, 2, 3, 4, 5, 6]
    pf = ParametrizedFunction(test_fun, θ)

    f = sensitivity_equation(pf, jacobian_buffer)

    y = zeros(2)
    dy = zeros(2, 6)
    x = [1, 2, 3]
    dx = Matrix{Float64}(reshape(1:18, 3, 6))
    k = 1
    f((y, dy), (x, dx))
    @test y == [22, 28]
    @test dy == [23.0 49.0 78.0 103.0 133.0 157.0;
        28.0 65.0 100.0 138.0 172.0 211.0]
    pf.θ .+= 2
    f((y, dy), (x, dx))
    @test y == [34, 40]
    @test dy == [35.0 79.0 126.0 169.0 217.0 259.0;
        40.0 95.0 148.0 204.0 256.0 313.0]

    f.θ .+= 2
    f((y, dy), (x, dx))
    @test y == [46, 52]
    @test dy == [47.0 109.0 174.0 235.0 301.0 361.0;
        52.0 125.0 196.0 270.0 340.0 415.0]
end
