list_procs = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
nprocess = maximum(list_procs)
addprocs(nprocess-1)
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
                                     list_procs)

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

@testset "Test gradient θ" begin
    function wrapper_cost(θ1)
        ms.new_simulation!(multiple_shoot, x0_list, θ1)
        return ms.cost_function(multiple_shoot)
    end

    function wrapper_gradient(θ1)
        ms.new_simulation!(multiple_shoot, x0_list, θ1)
        gradθ = zeros(3)
        gradθ_remote = ms.deepcopy_everywhere(gradθ, 1:nprocess)
        ms.gradient!(gradθ, gradθ_remote, multiple_shoot, "θ")
        return gradθ
    end

    function finite_difference_gradient(θ)
        return Calculus.gradient(wrapper_cost, θ)
    end

    # OBS: finite difference approximation
    # fails when trying parameters in the
    # chaotic region
    for θ in ([3.21, 3.2, 1.00],
            [3.21, 1.2, 1.54],
            [2.56, 3.27, 1.54])
        @test wrapper_gradient(θ) ≈ finite_difference_gradient(θ) rtol=1e-4
    end
end

@testset "Test gradient x0" begin
    function wrapper_cost(x0_expanded)
        x0_list = [[x0] for x0 in x0_expanded]
        ms.new_simulation!(multiple_shoot, x0_list, θ)
        return ms.cost_function(multiple_shoot)
    end

    function wrapper_gradient(x0_expanded)
        x0_list = [[x0] for x0 in x0_expanded]
        ms.new_simulation!(multiple_shoot, x0_list, θ)
        gradx0 = ms.deepcopy_everywhere(zeros(1), ones(list_procs))
        gradx0_remote = ms.deepcopy_everywhere(zeros(1), list_procs)
        ms.gradient!(gradx0, gradx0_remote, multiple_shoot, "x0")
        gradx0_expanded = [g[1] for g in gradx0]
        return gradx0_expanded
    end

    function finite_difference_gradient(x0_expanded)
        return Calculus.gradient(wrapper_cost, x0_expanded)
    end

    # OBS: finite difference approximation
    # fails when trying parameters in the
    # chaotic region
    for x0_expanded in (collect(linspace(0.2, 0.3, 10)),
                        collect(linspace(0.1, 0.8, 10)))
        @test wrapper_gradient(x0_expanded) ≈ finite_difference_gradient(x0_expanded) rtol=1e-3
    end
end

@testset "Test initialize_instance_everywhere" begin
    mat = ones(10, 10)
    mat_remote = ms.deepcopy_everywhere(mat, 1:5)
    for i =1:5
        if i != 1
            @test isa(mat_remote[i], Future)
            @test mat_remote[i].where == i
        end
        @test fetch(mat_remote[i]) == mat
    end
end
