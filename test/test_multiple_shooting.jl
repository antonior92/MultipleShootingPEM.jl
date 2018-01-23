addprocs()
@everywhere import MultipleShootingPEM
ms = MultipleShootingPEM
#@testset "Test on logistic map" begin
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
multiple_shoot = ms.MultipleShooting(f, g, x0_list,
                                     y, k0_list, θ,
                                     [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

@testset "Test function evaluation" begin
    for m = 1:10
        ms.new_simulation!(multiple_shoot, x0_list, [3.78, 3.78, 1.00])
        ys1 = vcat(fetch(multiple_shoot.simulations[m]).ys...)
        ms.new_simulation!(multiple_shoot, x0_list, [3.78, 3.78, 2.00])
        ys2 = vcat(fetch(multiple_shoot.simulations[m]).ys...)
        @test ys1 ≈ y[grid[m]:grid[m+1]-1]
        @test 2*ys1 ≈ ys2
    end
end
@testset "Test cost funciton" begin
    ms.new_simulation!(multiple_shoot, x0_list, [3.78, 3.78, 1.00])
    @test ms.cost_function(multiple_shoot) ≈ 0.0
    ms.new_simulation!(multiple_shoot, x0_list, [3.78, 3.78, 2.00])
    @test ms.cost_function(multiple_shoot) ≈ sum(vcat(y...).^2)
end
