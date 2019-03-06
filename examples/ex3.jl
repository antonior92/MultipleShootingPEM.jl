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
srand(4)
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
y_list = [[yi] for yi in y[2:end]]

θ = [β0, γ0]
multiple_shoot = ms.MultipleShooting(g, h,
                                     x0_list,
                                     y_list,
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

# Check optimized resuts
function compute_MSE(; β=β0, γ=γ0)
    y_pred = zeros(size(y))
    copy!(y_pred, y)
    for i = 2:N+1
        y_pred[i] = prediction(y[i-1], u[i-1], v[i-1], β=β, γ=γ)
    end
    mse = sum((y - y_pred).^2)
end

srand(4)
n_experiments = 10
θ_list = Array{Float64}[]
cost_list = Float64[]
res_list = []
θ0_list = [[β0 + 0.001*randn(); γ0+ 0.001*randn()] for i = 1:n_experiments]
for θ0 in θ0_list
    σ = 0.002
    x0_list_modified = [zeros(1) for i = 1:M]
    for i = 1:M
        x0_list_modified[i] += x0_list[i] + σ*randn()
    end
    opt = ms.OptimizationProblem(g, h, x0_list_modified, y_list,
                                 k0_list, θ0)

    res = ms.solve(opt, options=Dict("gtol" => 1e-10,
                                     "xtol" => 1e-10,
                                     "maxiter" => 2000,
                                     "initial_tr_radius" => 0.01))
    θ_est = res["x"][1:2]
    cost_est = res["fun"]


    println(θ_est)
    push!(θ_list, θ_est)
    push!(cost_list, cost_est)
    push!(res_list, res)
end


MSE = [compute_MSE(β=θi[1], γ=θi[2])  for θi in θ_list]
