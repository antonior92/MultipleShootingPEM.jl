nprocess = 5
pids = addprocs(nprocess-1)
list_procs = [1; 1; pids[[1, 1, 2, 2, 3, 3, 4, 4]]]
@everywhere import MultipleShootingPEM
ms = MultipleShootingPEM
# Define logistic map
function logistic_f(x_next, x, k, θ)
    x_next .=  θ*x - θ*x.^2
    return
end
function logistic_g(y, x, k, θ)
    y .= x
    return
end
# Define initial conditions
x0 = [0.5]
x0_list = Vector{Vector{Float64}}(10)
x0_list[1] = copy(x0)
grid = collect(1:10:101)
k0_list = grid[1:end-1]
# Initialize buffer
y = Vector{Vector{Float64}}(100)
for i = 1:length(y)
    y[i] = zeros(1)
end
# Simulate
x_final = copy(x0)
for i = 1:length(grid)-1
    x_final = ms.simulate_space_state!(y[grid[i]:grid[i+1]-1],
                                       logistic_f, logistic_g,
                                       x0, (grid[i], grid[i+1]-1), (3.78,))
    if i != length(grid)-1
        x0 .= x_final
        x0_list[i+1] = copy(x0)
    end
end
# Sanity check: Check if multiple simulations corresponds to a single one
y2 = Vector{Vector{Float64}}(100)
for i = 1:length(y)
    y2[i] = zeros(1)
end
x_final2 = ms.simulate_space_state!(y2, logistic_f, logistic_g,
                                   [0.5], (1, 100), (3.78,))
@test vcat(y...) == vcat(y2...)


# Simulate using multiple
θ = [3.78, 3.78, 1.00]
# Define polynomial model
# f(x) = θ1*x - θ2*x^2
@everywhere function f(y, dx, dθ, x, k, θ)
    y .= θ[1]*x - θ[2]*x.^2
    dx .= θ[1] - 2*θ[2]*x
    dθ .=  [ x  -x.^2  0]
    return
end
# g(x) = x
@everywhere function g(y, dx, dθ, x, k, θ)
    y .= θ[3]*x
    dx .= θ[3]
    dθ .=  [ 0 0 x ]
    return
end
opt = ms.OptimizationProblem(f, g, x0_list,
                              y, k0_list, θ,
                              list_procs)

@testset "Test cost funciton" begin
  θ_ext = [3.78; 3.78; 1.00; vcat(x0_list...)]
  @test ms.cost_function(opt, θ_ext) ≈ 0.0
  θ_ext = [3.78; 3.78; 2.00; vcat(x0_list...)]
  @test ms.cost_function(opt, θ_ext) ≈ sum(vcat(y...).^2)
end

@testset "Test gradient" begin
    function wrapper_cost(θ_ext)
        return ms.cost_function(opt, θ_ext)
    end

    function wrapper_gradient(θ_ext)
        return ms.gradient(opt, θ_ext)
    end

    function finite_difference_gradient(θ_ext)
        return Calculus.gradient(wrapper_cost, θ_ext, :central)
    end

    for θ in ([3.21, 3.2, 1.00],
            [3.21, 1.2, 1.54],
            [3.56, 3.07, 1.54])
        for x0_ext in (collect(linspace(0.2, 0.3, 10)),
                       collect(linspace(0.1, 0.8, 10)))
            θ_ext = [θ; x0_ext]
            @test isapprox(wrapper_gradient(θ_ext),
                           finite_difference_gradient(θ_ext),
                           rtol=1e-5)
        end
    end
end

@testset "Test Hessian" begin
    θ = [3.21, 3.2, 1.00]
    x0_ext = collect(linspace(0.2, 0.3, 10))
    θ_ext = [θ; x0_ext]
    hess = ms.hessian(opt, θ_ext)
    p = ones(13)
    @test length(hess["dot"](p)) == 13
end

@testset "Test constraints" begin
    θ_ext = [3.78; 3.78; 1.00; vcat(x0_list...)]
    @test ms.constraint(opt, θ_ext) ≈ zeros(9)
    θ_ext = [3.78; 3.78; 2.00; vcat(x0_list...)]
    @test ms.constraint(opt, θ_ext) ≈ zeros(9)
end
rmprocs(pids)

@testset "Test extended vector conversions" begin
    srand(1)
    Nθ = 10
    Nx = 3
    M = 5
    θ = randn(Nθ)
    x0_list = Vector{Vector{Float64}}(M)
    for i = 1:M
        x0_list[i] = randn(Nx)
    end
    θ_extended = [θ; vcat(x0_list...)]
    @testset "Test build extended vector" begin
        θ_extended_buffer = zeros(θ_extended)
        ms.build_extended_vector!(θ_extended_buffer, θ, x0_list, Nθ, M, Nx)
        @test θ_extended_buffer ≈ θ_extended
    end

    @testset "Test read extended vector" begin
        θ_buffer = zeros(θ)
        x0_list_buffer = Vector{Vector{Float64}}(M)
        for i = 1:M
            x0_list_buffer[i] = zeros(Nx)
        end
        ms.read_extended_vector!(θ_buffer, x0_list_buffer, θ_extended, Nθ, M, Nx)
        @test θ_buffer ≈ θ
        @test x0_list_buffer ≈ x0_list
    end
end
