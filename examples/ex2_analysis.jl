using FileIO
import ParallelTrainingNN
using Plots
using StatPlots
import CSV
using DataFrames
plotlyjs()
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
seeds = 1:1
shoot_len_list = [711, 50, 20, 10, 5, 3]
θ_est = Matrix{Vector{Float64}}(length(seeds),  length(shoot_len_list))
MSE = Matrix{Float64}(length(seeds), length(shoot_len_list))
for (i, s) in enumerate(seeds)
    for (j, shoot_len) in enumerate(shoot_len_list)
        file_name = "solutions/sol"*string(s)*"_"*string(shoot_len)*".jld2"
        res = load(file_name, "res")
        θ_est[i, j] = res["x"][1:41]
        MSE[i, j] = mean((ys - nn.predict(freerun, θ_est[i, j])).^2 )
    end
end

# Plot solution to the first problem
nn.plot_output(validation_data, freerun, θ_est[1, :],
               label=["shoot_len=711,", "shoot_len=50,", "shoot_len=20,",
                      "shoot_len=10,", "shoot_len=5,", "shoot_len=3,"])

# Box Plot
p = plot()
df = CSV.read("examples/MSE_parallel_considered_harmfull.csv")
violin!(p, ["SP"], [df[:series_parallel]])
violin!(p, ["P"], [df[:parallel]],  yscale=:log10)
violin!(p, ["N"], [MSE[:, 1]],  yscale=:log10)
violin!(p, ["50"], [MSE[:, 2]],  yscale=:log10)
violin!(p, ["20"], [MSE[:, 3]],  yscale=:log10)
violin!(p, ["10"], [MSE[:, 4]],  yscale=:log10)
violin!(p, ["5"], [MSE[:, 5]],  yscale=:log10)
violin!(p, ["3"], [MSE[:, 6]],  yscale=:log10)
# MSE = 1489.5 for ARX model (look at the matlab script)
x = 0.-0.3:1:12
plot!(p, x, 1489.5*ones(length(x)), color=:black, lw=2, s=:dash)

# Empirical cumulative distribution function
x =1:41
plot(x, sort(df[:series_parallel][1:41]), label= "SP" , yscale=:log10)
plot!(x, sort(df[:parallel][1:41]), label= "P" , yscale=:log10)
plot!(x, sort(MSE[:, 1]), label= "N" , yscale=:log10)
plot!(x, sort(MSE[:, 2]), label= "50" , yscale=:log10)
plot!(x, sort(MSE[:, 3]), label= "20" , yscale=:log10)
plot!(x, sort(MSE[:, 4]), label= "10" , yscale=:log10)
plot!(x, sort(MSE[:, 5]), label= "5" , yscale=:log10)
plot!(x, sort(MSE[:, 6]), label= "3" , yscale=:log10)
