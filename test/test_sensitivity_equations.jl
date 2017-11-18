@testset "Test sensitivity equation" begin
    # Problem Dimensions:
    # θ - size (6,); x - size (3,); y - size (2,);
    # dy/dx - size (2, 3); dy/dθ - size(2, 6).
    function test_fun(y, dx, dθ, x, θ)
        # Compute y
        y[1] = θ[1]*x[1] + θ[3]*x[2] + θ[5]*x[3]
        y[2] = θ[2]*x[1] + θ[4]*x[2] + θ[6]*x[3]
        # Compute df/dx
        dx[1, 1] = θ[1]
        dx[2, 1] = θ[2]
        dx[1, 2] = θ[3]
        dx[2, 2] = θ[4]
        dx[1, 3] = θ[5]
        dx[2, 3] = θ[6]
        # Compute df/dθ
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

    f = sensitivity_equation(test_fun, jacobian_buffer)

    y = zeros(2)
    dydθ = zeros(2, 6)
    dydϕ = zeros(2, 6)
    x = [1, 2, 3]
    dxdθ = Matrix{Float64}(reshape(1:18, 3, 6))
    dxdϕ = Matrix{Float64}(reshape(1:18, 3, 6))
    k = 1
    y_extended = (y, dydθ, dydϕ)
    x_extended = (x, dxdθ, dxdϕ)
    @time f(y_extended, x_extended, θ)
    @test y == [22, 28]
    @test dydθ == [23.0 49.0 78.0 103.0 133.0 157.0;
                    28.0 65.0 100.0 138.0 172.0 211.0]
    @test dydϕ == [22.0 49.0 76.0 103.0 130.0 157.0;
                   28.0 64.0 100.0 136.0 172.0 208.0]
    θ .= θ + 2
    f(y_extended, x_extended, θ)
    @test y == [34, 40]
    @test dydθ == [35.0 79.0 126.0 169.0 217.0 259.0;
                   40.0 95.0 148.0 204.0 256.0 313.0]
    @test dydϕ == [34.0 79.0 124.0 169.0 214.0 259.0;
                   40.0 94.0 148.0 202.0 256.0 310.0]

    θ .= θ + 2
    f(y_extended, x_extended, θ)
    @test y == [46, 52]
    @test dydθ == [47.0 109.0 174.0 235.0 301.0 361.0;
                   52.0 125.0 196.0 270.0 340.0 415.0]
    @test dydϕ == [46.0 109.0 172.0 235.0 298.0 361.0;
                   52.0 124.0 196.0 268.0 340.0 412.0]
end
