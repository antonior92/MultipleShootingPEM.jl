using FileIO
import ParallelTrainingNN
using Plots
using StatPlots
import CSV
using DataFrames
pyplot()
nn = ParallelTrainingNN

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
freerun = nn.free_run_simulation(mdl, yterms, uterms, validation_data);
# Get data
ys = nn.get_slice(freerun, validation_data.y);

# Compute MSE
seeds = 1:100
shoot_len_list = [3]
θ_est = Matrix{Vector{Float64}}(length(seeds),  length(shoot_len_list))
MSE = Matrix{Float64}(length(seeds), length(shoot_len_list))
res = Matrix(length(seeds), length(shoot_len_list))
for (i, s) in enumerate(seeds)
    for (j, shoot_len) in enumerate(shoot_len_list)
        file_name = "solutions_maxiter_3000_only3/sol"*string(s)*"_"*string(shoot_len)*".jld2"
        res[i, j] = try
            load(file_name, "res")
        catch
            false
        end
        if isa(res[i, j], Dict)
            θ_est[i, j] = res[i, j]["x"][1:41]
            MSE[i, j] = mean((ys - nn.predict(freerun, θ_est[i, j])).^2 )
        end
    end
end

# Read MSE file
ptch = CSV.read("examples/MSE_parallel_considered_harmfull.csv")

# Empirical cumulative distribution function
oness = ones(length(seeds))
p2 = plot(ylim=[3e2, 2e5], xticks=10:10:100, ylabel="MSE",
          xlabel="n", legend=:topleft)
cdf = sort(ptch[:series_parallel])
plot!(p2, seeds, cdf, label= "NN ARX", yscale=:log10,
     fillrange=[oness, cdf],
     color=:green, fillalpha=0.1)
cdf = sort(ptch[:parallel])
plot!(p2, seeds, cdf, label= "NN OE - SS",
     fillrange=[oness, cdf],
     color=:blue, fillalpha=0.1)
cdf = sort(MSE[:, 1])
plot!(p2, seeds, cdf, label= "NN OE - MS", yscale=:log10,
      fillrange=[oness, cdf], color=:red, fillalpha=0.1)
plot!(p2, seeds, 1489.5*oness, color=:black, lw=2, s=:dash,
      label= "Linear ARX")

savefig("neuralnet_mse.pdf")


# Plot solution to the first problem
#nn.plot_output(validation_data, freerun, θ_est[17, :],
#               label=["shoot_len=711,", "shoot_len=50,", "shoot_len=20,",
#                      "shoot_len=10,", "shoot_len=5,", "shoot_len=3,"])

# Box Plot
#p = plot()
#boxplot!(p, ["SP"], [ptch[:series_parallel]])
#boxplot!(p, ["P"], [ptch[:parallel]],  yscale=:log10)
#boxplot!(p, ["N"], [MSE[:, 1]],  yscale=:log10)
#boxplot!(p, ["50"], [MSE[:, 2]],  yscale=:log10)
#boxplot!(p, ["20"], [MSE[:, 3]],  yscale=:log10)
#boxplot!(p, ["10"], [MSE[:, 4]],  yscale=:log10)
#boxplot!(p, ["5"], [MSE[:, 5]],  yscale=:log10)
#boxplot!(p, ["3"], [MSE[:, 6]],  yscale=:log10)
# MSE = 1489.5 for ARX model (look at the matlab script)
#x = 0.-0.3:1:12
#plot!(p, x, 1489.5*ones(length(x)), color=:black, lw=2, s=:dash)
#ylims!(p, 100, 1000000)


# Scatter plot
#attribute = "execution_time"
#value = [[res[i, j][attribute] for i in 1:length(seeds)] for j in 1:length(shoot_len_list)]
#scatter(MSE, value, lab=["N" "50" "20" "10" "5" "3"], xlabel="MSE",
#         ylabel=attribute)
#df_1 = DataFrame(Type=CategoricalArray(repeat(map(string, shoot_len_list), inner=length(seeds))),
#                 MSE=MSE[:])
#
#df_2 = DataFrame(Type=CategoricalArray(repeat(["SP"], 100)),
#                 MSE=ptch[:series_parallel])
#df_3 = DataFrame(Type=CategoricalArray(repeat(["P"], 100)),
#                 MSE=ptch[:parallel])
#df = [df_1; df_2; df_3]
