using FileIO
using Plots
include("ex3_base.jl")
import Example3
pyplot()

# Load input
results = load("examples/ex3/solutions/sol_2019-04-05T17:03:43-341_18583/sim_len_40/results.jld2")["results"]
# Plot input
p_input = plot(results["data"]["u"], legend=false)
# Plot output
p_output = plot(results["data"]["y"], legend=false)
# Plot contour
p_contour = contour(results["grid"]["gl"], results["grid"]["ka"], results["grid"]["cost"],
                    fill=true, color=:viridis, levels=10)
scatter!(p_contour, [Example3.GL], [Example3.KA], color="red", legend=false)

if length(results["solutions"]) != 0
    g_l_list = [min(max(θ[1], results["options"]["grid_cost_funtion"][:gl][1]), results["options"]["grid_cost_funtion"][:gl][2]) for θ in results["solutions"]["theta"]]
    ka_list = [min(max(θ[2], results["options"]["grid_cost_funtion"][:ka][1]), results["options"]["grid_cost_funtion"][:ka][2]) for θ in results["solutions"]["theta"]]
    scatter!(p_contour, g_l_list', ka_list', color="blue")
    plot!(p_contour, legend=false)
end

plot!(p_contour)
xlims!(p_contour, (results["options"]["grid_cost_funtion"][:gl][1]-1,
                   results["options"]["grid_cost_funtion"][:gl][2]+1))
ylims!(p_contour, (results["options"]["grid_cost_funtion"][:ka][1]-0.1,
                   results["options"]["grid_cost_funtion"][:ka][2]+0.1))
