using FileIO
import ParallelTrainingNN
using Plots
using StatPlots
import CSV
using DataFrames
using LaTeXStrings
pgfplots()
nn = ParallelTrainingNN

# Import data
df = CSV.read("processed_data/training_set.csv");
ti, ui, yi = Vector(df[:time]), Vector(df[:input]), Vector(df[:output]);
Ts = (ti[2] - ti[1])/3600;
identification_data = nn.IdData(yi, ui, Ts);
df = CSV.read("processed_data/test_set.csv");
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
execution_time = Matrix{Float64}(length(seeds), length(shoot_len_list))
nfev = Matrix{Float64}(length(seeds), length(shoot_len_list))
res = Matrix(length(seeds), length(shoot_len_list))
for (i, s) in enumerate(seeds)
    for (j, shoot_len) in enumerate(shoot_len_list)
        file_name = "solutions/sol"*string(s)*"_"*string(shoot_len)*".jld2"
        res[i, j] = try
            load(file_name, "res")
        catch
            false
        end
        if isa(res[i, j], Dict)
            θ_est[i, j] = res[i, j]["x"][1:41]
            execution_time[i, j] = res[i, j]["execution_time"]
            nfev[i, j] = res[i, j]["nfev"]
            MSE[i, j] = mean((ys - nn.predict(freerun, θ_est[i, j])).^2 )
        end
    end
end

# Read MSE file
ptch = CSV.read("MSE_parallel_considered_harmfull.csv")

# Empirical cumulative distribution function
prob = 0.01:0.01:1
mse_min = 3e2
mse_max = 2e5
a = 0.05
n = length(seeds)
c = sqrt(1/(2n)*log(2/(1-a)))
ARXmse = 1489.5
# add some points at the extremity to make it easier to vizualize
#prob = vcat(0, prob, 1)
oness = ones(length(prob))
p2 = plot(xlim=[mse_min, mse_max], ylims = [0, 1], yticks=0:0.1:1,
          xlabel="MSE", legend=:bottomright)
cdf = sort(ptch[:series_parallel])
plot!(p2, vcat(mse_min, cdf, mse_max),
      vcat(0, prob, 1),
      label= "NN ARX",
      xscale=:log10,
      color=:green)
plot!(p2, cdf, [prob, prob], fillrange=[max.(prob-c, 0), min.(prob+c, 1)],
     fillalpha=0.1, label= "", xscale=:log10, color=:green)
cdf = sort(ptch[:parallel])
plot!(p2,vcat(mse_min, cdf, mse_max),
      vcat(0, prob, 1),
      label= "NN OE - SS",
     color=:blue)
plot!(p2, cdf, [prob, prob], fillrange=[max.(prob-c, 0), min.(prob+c, 1)],
      fillalpha=0.1, label= "", xscale=:log10, color=:blue)
plot!(p2, cdf[cdf .> ARXmse],
      prob[cdf .> ARXmse],
      label= "",
      xscale=:log10,
      marker=:circle,
      color=:blue)
cdf = sort(MSE[:, 1])
plot!(p2, vcat(mse_min, cdf, mse_max),
      vcat(0, prob, 1),
      label= "NN OE - MS", xscale=:log10,
      color=:red)
plot!(p2, cdf, [prob, prob], fillrange=[max.(prob-c, 0), min.(prob+c, 1)],
      fillalpha=0.1, label= "", xscale=:log10, color=:red)
plot!(p2, cdf[cdf .> ARXmse],
      prob[cdf .> ARXmse],
      label= "",
      xscale=:log10,
      marker=:circle,
      color=:red)
plot!(p2, 1489.5*oness, prob, color=:black, lw=1, s=:dash,
      label= "Linear ARX")

savefig("neuralnet_mse.tex")

println(mean(execution_time[:, 1]));
println(mean(nfev[:, 1]));
