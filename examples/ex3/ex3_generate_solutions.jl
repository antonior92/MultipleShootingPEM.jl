include("ex3_base.jl")
import Example3
using JLD2
using Plots
using JSON

options_dict = Dict("data_generator" => Dict(:t=>"pendulum",
                                             :N=>1000,
                                             :σw=>0,
                                             :σv=>0,
                                             :ampl=>10,
                                             :rep=>20,
                                             :seed=>1),
                    "pem" => Dict(:model=>"output_error",
                                  :sim_len=>20),
                    "grid_cost_funtion" => Dict(),
                    "solve_grid" => Dict(:gl=>(10, 60),
                                         :ka=>(0.5, 10),
                                         :npoints => (0, 2)))
results = Example3.runner(options_dict)


time = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS-sss")
mkdir("examples/ex3/solutions/sol_"*time)
# Save all data
JLD2.@save "examples/ex3/solutions/sol_"*time*"/results.jld2" results
# Plot figures
p_input = plot(results["data"]["u"])
savefig(p_input, "examples/ex3/solutions/sol_"*time*"/input_signal.png")
p_output = plot(results["data"]["y"])
savefig(p_output, "examples/ex3/solutions/sol_"*time*"/output_signal.png")
function plot_contour(results; log=false)
    if log
        p = contour(results["grid"]["gl"], results["grid"]["ka"], log10(results["grid"]["cost"]),
                 fill=true, color=:viridis, levels=10)
    else
        p = contour(results["grid"]["gl"], results["grid"]["ka"], results["grid"]["cost"],
                 fill=true, color=:viridis, levels=10)
    end
    scatter!(p, [Example3.GL], [Example3.KA], color="red")
    if length(results["solutions"]) != 0
        g_l_list = [θ[1] for θ in results["solutions"]["theta"]]
        ka_list = [θ[2] for θ in results["solutions"]["theta"]]
        scatter!(p, g_l_list', ka_list', color="blue")
        plot!(p, legend=false)
    end
    return p
end
p_contour = plot_contour(results, log=false)
savefig(p_contour, "examples/ex3/solutions/sol_"*time*"/contour.png")
p_contour = plot_contour(results, log=true)
savefig(p_contour, "examples/ex3/solutions/sol_"*time*"/contour_log.png")
if length(results["solutions"]) != 0
    p_hist = histogram(results["solutions"]["cost"])
    savefig(p_hist, "examples/ex3/solutions/sol_"*time*"/histogram.png")
end
# Save config file separately to be easy to check it
json_string = JSON.json(results["options"])
open("examples/ex3/solutions/sol_"*time*"/config.json","w") do f
    write(f, json_string)
end
