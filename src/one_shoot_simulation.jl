struct OneShootSimulation{T, N, Ny, Nx, Nθ}
    f::Function
    g::Function
    x0::T
    y::Vector{T}
    time_span::Tuple{Int, Int}
    # Sensitivity equations
    Jf::Function
    Jg::Function
    # x
    x
    dxdθ
    dxdx0
    x_extended
    # x_buffer
    x_buffer
    dxdθ_buffer
    dxdx0_buffer
    x_buffer_extended
    # Simulated y
    ys
    dydθ
    dydx0
    ys_extended
    # Buffer y
    y_buffer
end

function OneShootSimulation{T}(f::Function, g::Function,
                               x0::T, y::Vector{T}, k0::Int, θ)
    # Get sizes
    N = length(y)
    Ny = length(y[1])
    Nx = length(x0)
    Nθ = length(θ)
    time_span = (k0, k0+N-1)
    # Define x0_extended
    x = deepcopy(x0)
    dxdθ = zeros(Nx, Nθ)
    dxdx0 = eye(Nx, Nx)
    x_extended = (x, dxdθ, dxdx0)
    # Define x_buffer
    x_buffer = deepcopy(x0)
    dxdθ_buffer = zeros(Nx, Nθ)
    dxdx0_buffer = eye(Nx, Nx)
    x_buffer_extended = (x_buffer, dxdθ_buffer, dxdx0_buffer)
    # define ys extended
    ys_extended = Vector{Tuple{T, Matrix{Float64},Matrix{Float64}}}(N)
    ys = Vector{T}(N)
    dydθ = Vector{Matrix{Float64}}(N)
    dydx0 = Vector{Matrix{Float64}}(N)
    for i = 1:N
        ys[i] = zeros(Ny)
        dydθ[i] = zeros(Ny, Nθ)
        dydx0[i] = eye(Ny, Nx)
        ys_extended[i] = (ys[i], dydθ[i], dydx0[i])
    end
    # Define auxiliar buffers
    buffer_Jf = zeros(Nx, Nx)
    buffer_Jg = zeros(Ny, Nx)
    y_buffer = zeros(Ny)
    # Sensitivity equations
    Jf = sensitivity_equation(f, buffer_Jf)
    Jg = sensitivity_equation(g, buffer_Jg)
    # Simulate
    simulate_space_state!(ys_extended, Jf, Jg, x_extended,
                          time_span, (θ,); x0_2=x_buffer_extended)
    OneShootSimulation{T, N, Ny, Nx, Nθ}(
        f, g, deepcopy(x0), y, time_span, Jf, Jg, x, dxdθ, dxdx0, x_extended,
        x_buffer, dxdθ_buffer, dxdx0_buffer, x_buffer_extended,
        ys, dydθ, dydx0, ys_extended, y_buffer)
end

function new_simulation!{T, N, Ny, Nx, Nθ}(
        oss::OneShootSimulation{T, N, Ny, Nx, Nθ}, x0::T, θ)
    # Define x0_extended
    copy!(oss.x0, x0)
    copy!(oss.x, x0)
    copy!(oss.dxdθ, zeros(Nx, Nθ))
    copy!(oss.dxdx0, eye(Nx, Nx))
    # Define x_buffer
    copy!(oss.x_buffer, x0)
    copy!(oss.dxdθ_buffer, zeros(Nx, Nθ))
    copy!(oss.dxdx0_buffer, eye(Nx, Nx))
    # Simulate
    simulate_space_state!(oss.ys_extended, oss.Jf, oss.Jg, oss.x_extended,
                          oss.time_span, (θ,);
                          x0_2=oss.x_buffer_extended)
    return oss
end

function cost_function{T, N, Ny, Nx, Nθ}(
        oss::OneShootSimulation{T, N, Ny, Nx, Nθ},
        loss=L2DistLoss())
    cost = 0
    for i = 1:N
        cost += value(loss, oss.y[i], oss.ys[i], AvgMode.Sum())
    end
    return cost
end

function gradient!{T, N, Ny, Nx, Nθ}(grad::Vector{Float64},
        oss::OneShootSimulation{T, N, Ny, Nx, Nθ},
        variable="θ", loss=L2DistLoss(), accumulat=false)
    if !accumulat
        fill!(grad, 0)
    end
    if variable == "θ"
        J = oss.dydθ
    else
        J = oss.dydx0
    end

    for i = 1:N
        deriv!(oss.y_buffer, loss, oss.y[i], oss.ys[i])
        Base.LinAlg.BLAS.gemv!('T', 1.0, J[i], oss.y_buffer, 1.0, grad)
    end
    return grad
end

function hessian_approx!{T, N, Ny, Nx, Nθ}(hessp::Vector{Float64},
        oss::OneShootSimulation{T, N, Ny, Nx, Nθ}, p::Vector{Float64},
        variable="θ", loss=L2DistLoss(), accumulat=false)
    if !accumulat
        fill!(hessp, 0)
    end
    if variable == "θ"
        J = oss.dydθ
    else
        J = oss.dydx0
    end

    for i = 1:N
        A_mul_B!(oss.y_buffer, J[i], p)
        oss.y_buffer .*= deriv2.(loss, oss.y[i], oss.ys[i])
        Base.LinAlg.BLAS.gemv!('T', 1.0, J[i], oss.y_buffer, 1.0, hessp)
    end
    return hessp
end

# Get functions (usefull when using julia native parallel functions)
get_x0(oss::OneShootSimulation) = oss.x0
get_x(oss::OneShootSimulation) = oss.x
get_dxdx0(oss::OneShootSimulation) = oss.dxdx0
get_dxdθ(oss::OneShootSimulation) = oss.dxdθ
