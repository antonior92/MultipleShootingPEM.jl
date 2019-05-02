using FileIO
using Plots
include("ex3_base.jl")
import Example3
using LaTeXStrings
using CSV, DataFrames
pgfplots()

solutions = "solutions/"

function contour_plot(results; use_log=false)
    levels = 10
    if !use_log
        z = results["grid"]["cost"]
    else
        z = log10.(results["grid"]["cost"])
        if results["options"]["pem"][:sim_len] == 1024
            levels = 0:0.25:7
        elseif results["options"]["pem"][:sim_len] == 32
            levels = -4:0.25:2
        end
    end
    p_contour = contourf(results["grid"]["gl"], results["grid"]["ka"], z,
                        color=:viridis, levels=levels, xguide=L"g/l", yguide=L"k_a")
    scatter!(p_contour, [Example3.GL], [Example3.KA], color="red", legend=false, ms=8)
    if length(results["solutions"]) != 0
        g_l_list = [min(max(θ[1], results["options"]["grid_cost_funtion"][:gl][1]), results["options"]["grid_cost_funtion"][:gl][2]) for θ in results["solutions"]["theta"]]
        ka_list = [min(max(θ[2], results["options"]["grid_cost_funtion"][:ka][1]), results["options"]["grid_cost_funtion"][:ka][2]) for θ in results["solutions"]["theta"]]
        scatter!(p_contour, g_l_list', ka_list', color="blue", ms=8)
        plot!(p_contour, legend=false)
    end
    return p_contour
end

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
        p_contour = contour_plot(results, use_log=false)
        savefig(p_contour, joinpath(dir, "countour.pdf"))
        savefig(p_contour, joinpath(dir, "countour.png"))
        # Plot log contour
        p_contourl = contour_plot(results, use_log=true)
        savefig(p_contourl, joinpath(dir, "countour_log.pdf"))
        savefig(p_contourl, joinpath(dir, "countour_log.png"))
        pgfplots()
        p_contourl = contour_plot(results, use_log=true)
        savefig(p_contourl, joinpath(dir, "countour_log.tex"))
        # Save solutions
        CSV.write(joinpath(dir, "solutions.csv"),
                  DataFrame(hcat(results["solutions"]["theta"]...)'),
                  header=false)
       # Save grid
       ka = repeat(Array(results["grid"]["ka"])[1:2:end], outer=50)
       gl = repeat(Array(results["grid"]["gl"])[1:2:end], inner=50)
       cost = reshape(results["grid"]["cost"][1:2:end, 1:2:end]', (2500,))
       CSV.write(joinpath(dir, "grid.csv"),
                 DataFrame(hcat(gl, ka, cost)),
                 header=false)

    end
end
