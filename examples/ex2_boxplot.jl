addprocs(8)
@everywhere import MultipleShootingPEM
@everywhere ms = MultipleShootingPEM
@everywhere import ParallelTrainingNN
@everywhere nn = ParallelTrainingNN
@everywhere using Plots
pgfplots()
using LaTeXStrings
@everywhere using CSV

# Import data
@everywhere df = CSV.read("examples/processed_data/training_set.csv");
@everywhere ti, ui, yi = Vector(df[:time]), Vector(df[:input]), Vector(df[:output]);
@everywhere Ts = (ti[2] - ti[1])/3600;
@everywhere identification_data = nn.IdData(yi, ui, Ts);
@everywhere df = CSV.read("examples/processed_data/test_set.csv");
@everywhere tv, uv, yv = Vector(df[:time]), Vector(df[:input]), Vector(df[:output]);
@everywhere Ts = (tv[2] - tv[1])/3600;
@everywhere validation_data = nn.IdData(yv, uv, Ts);

# Define neural network model
@everywhere  mdl = nn.FeedforwardNetwork(2, 1, [10])
@everywhere yterms = [[1]];
@everywhere uterms = [[1]];
@everywhere mdl = nn.learn_normalization(mdl, yterms, uterms, identification_data);
@everywhere θ0 = nn.initial_guess(mdl)
@everywhere dx_aux = Matrix{Float64}(1, 2)
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
@everywhere function compute_results(seed)
    print(seed)
    M_list = [1, 60]
    actual_M_list = zeros(M_list)
    res_list = Vector(length(M_list))
    for (i, M) in enumerate(M_list)
        print(M)
        srand(seed)
        k0 = 1
        N = length(yi)
        k0_list = collect(k0:Int(floor(N//M)):k0+N-1)
        list_procs = ones(Int, length(k0_list))

        yi_aux = [[element] for element in yi]
        x0_list = yi_aux[k0_list]
        opt = ms.OptimizationProblem(f, g, x0_list, yi_aux, k0_list, θ0,
                                                 list_procs)

        res = ms.solve(opt, options=Dict("gtol" => 1e-10,
                                         "xtol" => 1e-10,
                                         "maxiter" => 2000,
                                         "initial_trust_radius" => 1))
        res_list[i] = res
        actual_M_list[i] = length(k0_list)
    end
    return res_list
end

ansert = pmap(compute_results, 1:15)
