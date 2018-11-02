import MultipleShootingPEM
ms = MultipleShootingPEM
using Plots
pyplot()
using LaTeXStrings

# Generate Data
k0 = 1
N = 200
θ_opt = 3.78
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
y = Vector{Vector{Float64}}(N)
for i = 1:length(y)
    y[i] = zeros(1)
end
x_final = ms.simulate_space_state!(y, logistic_f, logistic_g,
                                   x0, (k0, k0+N-1), (θ_opt,))

# Multiple shooting error estimation
M = 100
k0_list = collect(k0:Int(N//M):k0+N-1)
x0_list = y[k0_list]
list_procs = ones(Int, M)

# Define polynomial model
# f(x) = θ1*x - θ1*x^2
@everywhere function f(y, dx, dθ, x, k, θ)
    y .= θ[1]*x - θ[1]*x.^2
    dx .= θ[1] - 2*θ[1]*x
    dθ .=  x-x.^2
    return
end
# g(x) = x
@everywhere function g(y, dx, dθ, x, k, θ)
    y .= x
    dx .= 1
    dθ .= 0
    return
end

θ = 0
multiple_shoot = ms.MultipleShooting(f, g, x0_list,
                                     y, k0_list, θ,
                                     list_procs)
ms.cost_function(multiple_shoot)

# Get cost function
θ_min = 2.0
θ_delta = 0.01
θ_max = 4.0
θ_range = θ_min:θ_delta:θ_max
npoints = length(θ_range)
cost = Vector{Float64}(npoints)
i = 1
for θ = θ_range
    ms.new_simulation!(multiple_shoot, x0_list, [θ])
    cost[i] = ms.cost_function(multiple_shoot)
    i += 1
end

nrealiz = 20
cost_matrix = Matrix{Float64}(npoints, nrealiz)
srand(1)
j = 1
for j = 1:nrealiz
    σ = 0.005
    x0_list_modified = copy(x0_list)
    for i = 1:M
        x0_list_modified .+= σ*randn()
    end
    i = 1
    for θ = θ_range
        ms.new_simulation!(multiple_shoot, x0_list_modified, [θ])
        cost_matrix[i, j] = ms.cost_function(multiple_shoot)
        i += 1
    end
    j += 1
end



# Get solver solution
srand(4)
θ_list = Float64[]
cost_list = Float64[]
res_list = []
for θp = 3.2:0.05:3.9
    σ = 0.002
    x0_list_modified = copy(x0_list)
    for i = 1:M
        x0_list_modified .+= σ*randn()
    end
    θ0 = [θp]
    opt = ms.OptimizationProblem(f, g, x0_list_modified, y,
                                 k0_list, θ0, list_procs)

    res = ms.solve(opt, options=Dict("gtol" => 1e-10,
                                     "xtol" => 1e-10,
                                     "maxiter" => 2000))
    θ_est = res["x"][1]
    cost_est = res["fun"]

    push!(θ_list, θ_est)
    push!(cost_list, cost_est)
    push!(res_list, res)
end


plot(θ_range, cost, color=:black, lw=2, legend=false, grid=false,
     xlims=[θ_min, θ_max], xguide=L"\theta", yguide=L"V(\theta)")
plot!(θ_range, cost_matrix, color=:blue, linealpha=0.2)
scatter!(θ_list, cost_list, marker=:c, color=:green, markersize=5)
vline!([θ_opt], ls=:dot, color=:red)

#savefig("ratio2_ms_logistic.tex")

# Evaluate solver performance
niter_list = [res["niter"] for res in res_list]
median(niter_list)
maximum(niter_list)
minimum(niter_list)

nfun_list = [res["nfev"] for res in res_list]
median(nfun_list)
maximum(nfun_list)
minimum(nfun_list)

exect_list = [res["execution_time"] for res in res_list]
median(exect_list)
maximum(exect_list)
minimum(exect_list)
