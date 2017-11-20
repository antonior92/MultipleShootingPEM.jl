#= Imports =#
using Base.Test
include("../src/sensitivity_equations.jl")
include("../src/space_state_model.jl")
include("../src/one_shoot_simulation.jl")

#= Generate data =#
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
x_final = simulate_space_state!(y, logistic_f, logistic_g, [0.5], (1, 100), (3.78,))
x_final2 = simulate_space_state!(y2, logistic_f, logistic_g, [0.5], (1, 100), (3.62,))


#= One Shoot Simulation =#
# Define initial values
k0 = 1
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
   dθ .=  [ 1 1 x ]
   return
end

oss = OneShootSimulation(y, f, g, x0, k0, θ)

@test vcat(oss.ys...) - vcat(y...)  ≈ zeros(100)
@test oss.x ≈ x_final

new_simulation(oss, x0, θ2)

@test vcat(oss.ys...) - vcat(y2...)  ≈ zeros(100)
@test oss.x ≈ x_final2
