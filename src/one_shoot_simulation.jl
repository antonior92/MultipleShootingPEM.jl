struct OneShootSimulation{T, N, Ny, Nx, Nθ}
    y::Vector{T}
    f::Function
    g::Function
    x0::T
    k0::Int
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
end

function OneShootSimulation{T}(y::Vector{T}, f::Function, g::Function,
                               x0::T, k0::Int, θ)
    # Get sizes
    N = length(y); Ny = length(y[1]); Nx = length(x0); Nθ = length(θ);
    # Define x0_extended
    x = copy(x0)
    dxdθ = zeros(Nx, Nθ);
    dxdx0 = eye(Nx, Nx);
    x_extended = (x, dxdθ, dxdx0)
    # Define x_buffer
    x_buffer = copy(x0)
    dxdθ_buffer = zeros(Nx, Nθ);
    dxdx0_buffer = eye(Nx, Nx);
    x_buffer_extended = (x_buffer, dxdθ_buffer, dxdx0_buffer)
    # define ys extended
    ys_extended = Vector{Any}(N)
    ys = Vector{T}(N)
    dydθ = Vector{Matrix{Float64}}(N)
    dydx0 = Vector{Matrix{Float64}}(N)
    for i = 1:N
        ys[i] = zeros(y[i])
        dydθ[i] = zeros(Ny, Nθ)
        dydx0[i] = eye(Ny, Nx)
        ys_extended[i] = (ys[i], dydθ[i], dydx0[i])
    end
    # Define auxiliar buffers
    buffer_Jf = zeros(Nx, Nx); buffer_Jg = zeros(Ny, Nx);
    # Sensitivity equations
    Jf = sensitivity_equation(f, buffer_Jf)
    Jg = sensitivity_equation(g, buffer_Jg)
    # Simulate
    simulate_space_state!(ys_extended, Jf, Jg, x_extended,
                          (k0, k0+N-1), (θ,); x0_2=x_buffer_extended)
    OneShootSimulation{T, N, Ny, Nx, Nθ}(
        y, f, g, x0, k0, Jf, Jg, x, dxdθ, dxdx0, x_extended,
        x_buffer, dxdθ_buffer, dxdx0_buffer, x_buffer_extended, ys,
        dydθ, dydx0, ys_extended)
end

function new_simulation{T, N, Ny, Nx, Nθ}(
        oss::OneShootSimulation{T, N, Ny, Nx, Nθ}, x0::T, θ)
    # Define x0_extended
    copy!(oss.x, x0)
    copy!(oss.dxdθ, zeros(Nx, Nθ))
    copy!(oss.dxdx0, eye(Nx, Nx))
    # Define x_buffer
    copy!(oss.x_buffer, x0)
    copy!(oss.dxdθ_buffer, zeros(Nx, Nθ))
    copy!(oss.dxdx0_buffer, eye(Nx, Nx))
    # Simulate
    simulate_space_state!(oss.ys_extended, oss.Jf, oss.Jg, oss.x_extended,
                          (oss.k0, oss.k0+N-1), (θ,);
                          x0_2=oss.x_buffer_extended)
    return
end
