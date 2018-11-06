import MultipleShootingPEM
ms = MultipleShootingPEM
import ParallelTrainingNN
nn = ParallelTrainingNN
using Plots
pyplot()
using LaTeXStrings
using CSV

# Import data
df = CSV.read("examples/processed_data/training_set.csv");
ti, ui, yi = Vector(df[:time]), Vector(df[:input]), Vector(df[:output]);
Ts = (ti[2] - ti[1])/3600;
identification_data = nn.IdData(yi, ui, Ts);
df = CSV.read("examples/processed_data/test_set.csv");
tv, uv, yv = Vector(df[:time]), Vector(df[:input]), Vector(df[:output]);
Ts = (tv[2] - tv[1])/3600;
validation_data = nn.IdData(yv, uv, Ts);

# Define neural network model
mdl = nn.FeedforwardNetwork(2, 1, [10])
yterms = [[1]];
uterms = [[1]];
mdl = nn.learn_normalization(mdl, yterms, uterms, identification_data);
θ0 = nn.initial_guess(mdl)
dx_aux = Matrix{Float64}(1, 2)
@everywhere function f(y, dx, dθ, x, k, θ)
    x_aux = [x[1], ui[k]]
    nn.evaluate!(mdl, x_aux, θ, y; dx=dx_aux, dΘ=dθ)
    dx[1, 1] = dx_aux[1, 1]
    return
end
@everywhere function g(y, dx, dθ, x, k, θ)
    y .= x
    dx .= 1
    dθ .= 0
    return
end

# Multiple shooting error estimation
M_list = [1, 5, 20, 60, 100]
actual_M_list = zeros(M_list)
θ_est = Vector{Vector{Float64}}(length(M_list))
res_list = Vector(length(M_list))
for (i, M) in enumerate(M_list)
    print(M)
    srand(1)
    k0 = 1
    N = length(yi)
    k0_list = collect(k0:Int(floor(N//M)):k0+N-1)
    list_procs = ones(Int, length(k0_list))

    yi_aux = [[element] for element in yi]
    x0_list = yi_aux[k0_list]
    opt = ms.OptimizationProblem(f, g, x0_list, yi_aux, k0_list, θ0)

    res = ms.solve(opt, options=Dict("gtol" => 1e-10,
                                     "xtol" => 1e-10,
                                     "maxiter" => 2000,
                                     "initial_tr_radius" => 1))
    res_list[i] = res
    actual_M_list[i] = length(k0_list)
    θ_est[i] = res["x"][1:41]
end

freerun = nn.free_run_simulation(mdl, yterms, uterms, validation_data);
nn.plot_output(validation_data, freerun, θ_est,
               label=["M=1,", "M=5,", "M=20,", "M=60,", "M=100,"])
# savefig("pilot_plant_ms.tex")


# Compute validation error
ys = nn.get_slice(freerun, validation_data.y);
MSE = Vector{Float64}(length(M_list))
for i = 1: length(M_list)
    MSE[i] = mean((ys - nn.predict(freerun, θ_est[i])).^2 )
end
