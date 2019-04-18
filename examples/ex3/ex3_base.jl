module Example3

import MultipleShootingPEM
ms = MultipleShootingPEM
using LaTeXStrings
using ProgressMeter
using JLD2


############### Nonlinear system ###############
GL = 9.8/0.3
KA = 2.
M = 3.
D = 0.01
function nl_sys(x, u; gl=GL, ka=KA, m=M, δ=D)
    x1 = x[1] + δ*x[2]
    x2 = -δ*gl*sin(x[1]) + (1 - δ*ka/m)*x[2] + δ*1/m*u[1]
    return [x1, x2]
end

function derivative_x(x, u; gl=GL, ka=KA, m=M, δ=D)
    J = [1 δ;-δ*gl*cos(x[1]) (1 - δ*ka/m)]
    return J
end

function derivative_gl(x, u; gl=GL, ka=KA, m=M, δ=D)
    J = [0; -δ*sin(x[1])]
end

function derivative_ka(x, u; gl=GL, ka=KA, m=M, δ=D)
    J = [0; -δ*1/m*x[2]]
end

################ Controller ###############
GAIN = 4000
ZERO1 = 0.99
ZERO2 = 0.98
POLE1 = 0.02
CTRL = [GAIN, -GAIN*(ZERO1+ZERO2), GAIN*ZERO1*ZERO2, (1+POLE1), -POLE1]


################ Data Generator ##############
# Autonomous system
function pendulum(;N=2000, σv=0.,σw=0., seed=1, ampl=0, rep=20)
    srand(seed)
    if ampl != 0
        u = repeat(ampl*randn(Int(N//rep)), inner=rep)
        N = length(u)
    else
        u = zeros(N)
    end
    x0 = [π/2, 0]
    y = zeros(N)
    x = zeros(N+1, 2)
    x[1, :] = x0
    v = σv*randn(N)
    w = σw*randn(N)
    for k in 1:N
        x[k+1, :] = nl_sys(x[k, :], u[k]+ w[k])
        y[k] = x[k+1, 1] + v[k]
    end
    return Dict("u" => u, "y" => y, "x" => x, "v"=> v, "w" => w)
end

function inverted_pendulum(;N=2000, σv=0.,σw=0., ampl=0, rep=20, seed=1, transient=10000)
    srand(seed)
    Δr = repeat(ampl*randn(Int((N+transient)//rep)), inner=rep)
    total_len = length(Δr)
    n = length(Δr)
    u = 10*ones(total_len)
    x0 = [π, 0]
    x = zeros(total_len, 2)
    x[4, :] = x0
    y = π * ones(total_len)
    r = π * ones(total_len) + Δr
    e = zeros(total_len)
    v = σv*randn(total_len)
    w = σw*randn(total_len)
    for k in 4:total_len-1
        x[k+1, :] = nl_sys(x[k, :], u[k]+ w[k])
        y[k] = x[k+1, 1] + v[k]
        e[k] = r[k] - y[k]
        u[k+1] = dot(CTRL, [e[k], e[k-1], e[k-2], u[k], u[k-1]])
    end
    u, y, x = u[transient:end], y[transient:end], x[transient:end, :]
    return Dict("u" => u, "y" => y, "x" => x, "v" => v,
                "w" => w, "e" => e, "r" => r)
end

function data_generator(; t="pendulum", kwargs...)
    if t == "pendulum"
        return pendulum(; kwargs...)
    elseif t == "inverted_pendulum"
        return inverted_pendulum(; kwargs...)
    end
end

################ Output error model ##############
function output_error_model(data_dict; sim_len=1)
    u = data_dict["u"]
    y = data_dict["y"]
    xx = data_dict["x"]
    k0 = 2
    function g(x_next, dx, dθ, x, k, θ)
        gl = θ[1]
        ka = θ[2]
        k = k + k0
        x_next .= nl_sys(x, u[k]; gl=gl, ka=ka)
        dx .= derivative_x(x, u; gl=gl, ka=ka)
        dθ[:, 1] .= derivative_gl(x, u[k]; gl=gl, ka=ka)
        dθ[:, 2] .= derivative_ka(x, u[k]; gl=gl, ka=ka)
        return
    end

    function h(z, dx, dθ, x, k, θ)
        z .= x[1]
        dx .= [1 0]
        dθ .= [0 0]
        return
    end

    N = length(y)
    k0_list = collect(1:sim_len:N-k0-1)
    x0_ideal_list = [xx[k, :] for k in k0_list+k0]
    x0_list = [[y[k-1], (y[k] - y[k-1])/D] for k in k0_list+k0]
    y_list = [[yi] for yi in y[k0:end-1]]
    θ = [0, 0]
    return (ms.MultipleShooting(g, h, x0_ideal_list, y_list, k0_list, θ),
            ms.OptimizationProblem(g, h, x0_list, y_list, k0_list, θ))
end

function pem(data_dict; model="output_error", kwargs...)
    if model=="output_error"
        return output_error_model(data_dict; kwargs...)
    end
end

################ Compute cost function on grid of points ##############
function grid_cost_funtion(multiple_shoot; gl=(10, 60), ka=(0.5, 10), npoints=(50, 50))
    n_experiments = npoints[1]*npoints[2]
    if n_experiments == 0
        return Dict()
    end
    g_l_range = linspace(gl[1],  gl[2], npoints[1])
    ka_range = linspace(ka[1], ka[2], npoints[2])
    cost = Matrix{Float64}(npoints...)
    i = 1
    p = Progress(n_experiments, 1)
    for g_l in g_l_range
        j = 1
        for ka in ka_range
            x0_list = [oss.x0 for oss in multiple_shoot.simulations]
            ms.new_simulation!(multiple_shoot, x0_list, [g_l, ka])
            cost[i, j] = ms.cost_function(multiple_shoot)
            next!(p)
            j += 1
        end
        i += 1
    end

    return Dict("gl" => g_l_range, "ka" => ka_range, "cost" => cost')
end

################ Solve optimization problem for grid of initial points ##############
function solve_grid(opt; gl=(10, 60), ka=(0.5, 10), npoints=(3, 3))
    n_experiments = npoints[1]*npoints[2]
    if n_experiments == 0
        return Dict()
    end
    g_l_range = linspace(gl[1],  gl[2], npoints[1])
    ka_range = linspace(ka[1], ka[2], npoints[2])
    θ_list = Array{Float64}[]
    cost_list = Float64[]
    res_list = []
    MSE_list = []
    θ0_list = [[g_l, ka] for g_l in g_l_range for ka in ka_range]
    p = Progress(n_experiments, 1)
    for θ0 in θ0_list
        x0_list = [oss.x0 for oss in opt.ms.simulations]
        θ_ext = zeros(2 + length(x0_list)*2)
        ms.build_extended_vector!(θ_ext,  θ0, x0_list, 2, length(x0_list), 2)
        ms.update_simulation!(opt, θ_ext)
        res = ms.solve(opt, options=Dict("gtol" => 1e-10,
                                         "xtol" => 1e-10,
                                         "maxiter" => 2000,
                                         "initial_tr_radius" => 0.01,
                                         "verbose" => 2))
        θ_est = res["x"][1:2]
        cost_est = res["fun"]
        res = delete!(res, "jac")

        push!(θ_list, θ_est)
        push!(cost_list, cost_est)
        push!(res_list, res)
        next!(p)
    end
    return Dict("theta"=> θ_list, "cost"=> cost_list, "info" => res_list)
end


############ Runner ##############
function runner(options_dict)
    println("Generating data...")
    data_dict = data_generator(;options_dict["data_generator"]...)
    println("Done!")

    println("Creating prediction model...")
    multiple_shoot, opt = pem(data_dict; options_dict["pem"]...)
    println("Done!")

    println("Computing cost function in a grid...")
    grid_cost_fun = grid_cost_funtion(multiple_shoot; options_dict["grid_cost_funtion"]...)
    println("Done!")

    println("Solving for diferent initial conditions")
    solutions =  solve_grid(opt; options_dict["solve_grid"]...)
    println("Done!")
    Dict("data" => data_dict, "options"=>options_dict,
         "grid" => grid_cost_fun, "solutions" => solutions)
end

end # module
