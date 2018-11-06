addprocs(8)
@everywhere import MultipleShootingPEM
@everywhere ms = MultipleShootingPEM
@everywhere import ParallelTrainingNN
@everywhere nn = ParallelTrainingNN
@everywhere using CSV
@everywhere using JLD2

# Import data
@everywhere df = CSV.read("examples/processed_data/training_set.csv");
@everywhere ti, ui, yi = Vector(df[:time]), Vector(df[:input]), Vector(df[:output]);
@everywhere Ts = (ti[2] - ti[1])/3600;
@everywhere Ni = length(yi)

@everywhere identification_data = nn.IdData(yi, ui, Ts);
@everywhere df = CSV.read("examples/processed_data/test_set.csv");
@everywhere tv, uv, yv = Vector(df[:time]), Vector(df[:input]), Vector(df[:output]);
@everywhere Ts = (tv[2] - tv[1])/3600;
@everywhere validation_data = nn.IdData(yv, uv, Ts);

# Define neural network model
@everywhere mdl = nn.FeedforwardNetwork(2, 1, [10])
@everywhere yterms = [[1]];
@everywhere uterms = [[1]];
@everywhere mdl = nn.learn_normalization(mdl, yterms, uterms, identification_data);
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

@everywhere function compute_solution(seed, shoot_len)
    println("shoot_len = " * string(shoot_len))
    srand(seed)
    θ0 = nn.initial_guess(mdl)
    k0 = 1
    k0_list = collect(k0:shoot_len:k0+Ni-1)
    list_procs = ones(Int, length(k0_list))
    yi_aux = [[element] for element in yi]
    x0_list = yi_aux[k0_list]
    opt = ms.OptimizationProblem(f, g, x0_list, yi_aux, k0_list, θ0,
                                list_procs)

    res = ms.solve(opt, options=Dict("gtol" => 1e-12,
                                     "xtol" => 1e-12,
                                     "maxiter" => 3000,
                                     "initial_tr_radius" => 1))
    delete!(res, "jac")
    JLD2.@save "solutions/sol"*string(seed)*"_"*string(shoot_len)*".jld2" res
end

# Multiple shooting error estimation
@everywhere function compute_solutions(seed)
    println("seed = " * string(seed))
    shoot_len_list = [Ni, 50, 20, 10, 5, 3]
    for shoot_len in shoot_len_list
        compute_solution(seed, shoot_len)
    end
end

if ~Base.Filesystem.isdir("solutions")
    Base.Filesystem.mkdir("solutions")
end

pmap(compute_solutions, 1:100)
