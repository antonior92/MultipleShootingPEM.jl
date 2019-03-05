import MultipleShootingPEM
ms = MultipleShootingPEM
using Plots
pyplot()
using LaTeXStrings

# Params
σ0 = 0.1
β0 = 10
γ0 = 8

# Noise free prediction
function prediction(y, u, v; β=β0, γ=γ0)
    return 1 - 2 * exp(-β * (y+ γ*u)^2 * v^2)
end

function derivative_β(y, u, v; β=β0, γ=γ0)
    return 2 * (y + γ*u)^2 * v^2 * exp(-β * (y+ γ*u)^2 * v^2)
end

function derivative_γ(y, u, v; β=β0, γ=γ0)
    return 4 * β * (y + γ*u) * u * v^2 * exp(-β * (y+ γ*u)^2 * v^2)
end

function derivative_v(y, u, v; β=β0, γ=γ0)
    return 4 * β * (y + γ*u)^2 * v * exp(-β * (y+ γ*u)^2 * v^2)
end

# Generate Data
N = 200
rep = 10
v = σ0*rand(N+1)
y = zeros(N+1)
srand(1)
u = repeat(rand(Int(N//rep)), inner=rep)
for i = 2:N+1
    y[i] = prediction(y[i-1], u[i-1], v[i-1]) + v[i]
end
plot(y)
plot!(u)

# Define prediction model
function g(x_next, dx, dθ, x, k, θ)
    β = θ[1]
    γ = θ[2]
    x_next[1] = y[k] - prediction(y[k], u[k], x[1], β=β, γ=γ)
    dx[1, 1] = - derivative_v(y[k], u[k], x[1], β=β, γ=γ)
    dθ[1, 1] = - derivative_β(y[k], u[k], x[1], β=β, γ=γ)
    dθ[1, 2] = - derivative_γ(y[k], u[k], x[1], β=β, γ=γ)
    return
end

function h(z, dx, dθ, x, k, θ)
    β = θ[1]
    γ = θ[2]
    z[1] = prediction(y[k], u[k], x[1], β=β, γ=γ)
    dx[1, 1] = derivative_v(y[k], u[k], x[1], β=β, γ=γ)
    dθ[1, 1] = derivative_β(y[k], u[k], x[1], β=β, γ=γ)
    dθ[1, 2] = derivative_γ(y[k], u[k], x[1], β=β, γ=γ)
    return
end

# Multiple shooting error estimation
M = 200
k0 = 1
k0_list = collect(k0:Int(N//M):k0+N-1)
x0_list =[[x_i] for x_i in v[k0_list+1]]
list_procs = ones(Int, M)

θ = [β0, γ0]
multiple_shoot = ms.MultipleShooting(g, h,
                                     x0_list,
                                     [[yi] for yi in y[2:end]],
                                     k0_list, θ)
ms.cost_function(multiple_shoot)

# Get cost function
β_range = 5:0.1:20
γ_range = 0:0.1:10
npoints = (length(β_range), length(γ_range))
cost = Matrix{Float64}(npoints...)
i = 1
for β = β_range
    j = 1
    for γ = γ_range
        ms.new_simulation!(multiple_shoot, x0_list, [β, γ])
        cost[i, j] = ms.cost_function(multiple_shoot)
        j += 1
    end
    i += 1
end

β_aux = repeat(reshape(β_range, 1, :), length(γ_range), 1)
γ_aux = repeat(γ_range, 1, length(β_range))
contour(β_aux, γ_aux, cost', fill=true, color=:viridis)
scatter!([β0], [γ0])
