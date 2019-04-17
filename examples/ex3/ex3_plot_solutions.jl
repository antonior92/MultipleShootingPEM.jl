using FileIO
using Plots
include("ex3_base.jl")
import Example3
using LaTeXStrings
pgfplots()

solutions = "solutions/"

for root_dir in take!(walkdir(solutions))[2]
    # Plot contour
    first = true
    for dir_aux in take!(walkdir(joinpath(solutions, root_dir)))[2]
        dir = joinpath(solutions, root_dir, dir_aux)
        results = load(joinpath(dir, "results.jld2"))["results"]
        if first
            # Plot input
            p_input = plot(results["data"]["u"], legend=false, xguide=L"k", yguide=L"u", grid=false)
            savefig(p_input, joinpath(solutions, root_dir, "input_signal.pdf"))
            savefig(p_input, joinpath(solutions, root_dir, "input_signal.tex"))
            # Plot output
            p_output = plot(results["data"]["y"]/π, legend=false, xguide=L"k", yguide=L"$\frac{y}{\pi}$",
                           grid=false)
            savefig(p_output, joinpath(solutions, root_dir, "output_signal.pdf"))
            savefig(p_output, joinpath(solutions, root_dir, "output_signal.tex"))
        end
        first=false
        # Plot countour
        pyplot()
        p_contour = contourf(results["grid"]["gl"], results["grid"]["ka"], results["grid"]["cost"],
                             color=:viridis, levels=10)
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
        savefig(p_contour, joinpath(dir, "countour.pdf"))
        # Plot log contour
        p_contourl = contourf(results["grid"]["gl"], results["grid"]["ka"], log10.(results["grid"]["cost"]),
                             color=:viridis, levels=10)
        scatter!(p_contourl, [Example3.GL], [Example3.KA], color="red", legend=false)

        if length(results["solutions"]) != 0
            g_l_list = [min(max(θ[1], results["options"]["grid_cost_funtion"][:gl][1]), results["options"]["grid_cost_funtion"][:gl][2]) for θ in results["solutions"]["theta"]]
            ka_list = [min(max(θ[2], results["options"]["grid_cost_funtion"][:ka][1]), results["options"]["grid_cost_funtion"][:ka][2]) for θ in results["solutions"]["theta"]]
            scatter!(p_contourl, g_l_list', ka_list', color="blue")
            plot!(p_contourl, legend=false)
        end

        plot!(p_contourl)
        xlims!(p_contourl, (results["options"]["grid_cost_funtion"][:gl][1]-1,
                           results["options"]["grid_cost_funtion"][:gl][2]+1))
        ylims!(p_contourl, (results["options"]["grid_cost_funtion"][:ka][1]-0.1,
                           results["options"]["grid_cost_funtion"][:ka][2]+0.1))
        savefig(p_contourl, joinpath(dir, "countour_log.pdf"))
        pgfplots()
    end
end

# Plot graphs
#pyplot()

#for sim_len in [2^i for i in 0:7]
    # Directory
#    dir = root_dir * "sim_len_" * string(sim_len) * "/"
    # Load input
#    results = load(dir*"results.jld2")["results"]
#end
