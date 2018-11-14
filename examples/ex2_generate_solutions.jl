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

@everywhere function compute_solution(seed, shoot_len::Int)
    println("shoot_len = " * string(shoot_len))
    srand(seed)
    θ0 = nn.initial_guess(mdl)
    k0 = 1
    k0_list = collect(k0:shoot_len:k0+Ni-1)
    yi_aux = [[element] for element in yi]
    x0_list = yi_aux[k0_list]
    opt = ms.OptimizationProblem(f, g, x0_list, yi_aux, k0_list, θ0)

    res = ms.solve(opt, options=Dict("gtol" => 1e-12,
                                     "xtol" => 1e-12,
                                     "maxiter" => 2000,
                                     "initial_constr_penalty" => 0.01))
    delete!(res, "jac")
    JLD2.@save "solutions/sol"*string(seed)*"_"*string(shoot_len)*".jld2" res
end

function compute_solution(seed, shoot_len::String)
    tic()
    if shoot_len == "ARX"
        dynfit, opt, e = nn.narx(mdl, yterms, uterms, identification_data,
                                 maxIter=2000);
    elseif shoot_len == "OE"
        dynfit, opt, e = nn.noe(mdl, yterms, uterms, identification_data,
                                use_extended=false, maxIter=2000);
    end
    execution_time = toc()
    res = Dict("x" => dynfit, "execution_time" => execution_time)
    JLD2.@save "solutions/sol"*string(seed)*"_"*string(shoot_len)*".jld2" res
end

# Multiple shooting error estimation
@everywhere function compute_solutions(seed)
    println("seed = " * string(seed))
    shoot_len_list = [3]
    for shoot_len in shoot_len_list
        compute_solution(seed, shoot_len)
    end
end


if ~Base.Filesystem.isdir("solutions")
    Base.Filesystem.mkdir("solutions")
end

#@everywhere compute_solution(shoot_len) = compute_solution(1, shoot_len)
#pmap(compute_solution, [Ni, 50, 20, 10, 5, 3])
pmap(compute_solutions_original_paper, 1:100)
