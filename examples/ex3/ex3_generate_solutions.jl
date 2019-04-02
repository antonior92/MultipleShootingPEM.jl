include("ex3_base.jl")
import Example3
using JLD2
using Plots
using JSON
pyplot()

N=1000
function run_and_save(options_dict)
    if ~isdir("examples/ex3/solutions")
        mkdir("examples/ex3/solutions")
    end
    time = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS-sss")
    root_dir = "examples/ex3/solutions/sol_"*time
    mkdir(root_dir)
    for sim_len in [1, 2, 3, 5, 10, 20, 40, N]
        options_dict["pem"] = merge(options_dict["pem"], Dict(:sim_len=>10))
        results = Example3.runner(options_dict)
        dir = root_dir*"/sim_len_"*string(sim_len)
        mkdir(dir)
        # Save all data
        JLD2.@save dir*"/results.jld2" results
        # Plot figures
        try
            p_input = plot(results["data"]["u"])
            savefig(p_input, dir*"/input_signal.png")
        end
        try
            p_output = plot(results["data"]["y"])
            savefig(p_output, dir*"/output_signal.png")
        end
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
        try
            p_contour = plot_contour(results, log=false)
            savefig(p_contour, dir*"/contour.png")
        end
        try
            p_contour = plot_contour(results, log=true)
            savefig(p_contour, dir*"/contour_log.png")
        end
        try
            if length(results["solutions"]) != 0
                p_hist = histogram(results["solutions"]["cost"])
                savefig(p_hist, dir*"/histogram.png")
            end
        end
        # Save config file separately to be easy to check it
        json_string = JSON.json(results["options"])
        open(dir*"/config.json","w") do f
            write(f, json_string)
        end
    end
end


options_dicts = []
for ampl in [0, 10, 30, 50]
    for σv in [0, 0.2]
        d = Dict("data_generator" => Dict(:t=>"pendulum", :N=>N,
                                          :σw=>0., :σv=>σv,
                                          :ampl=>ampl, :rep=>20,
                                          :seed=>1),
                 "pem" => Dict(:model=>"output_error"),
                 "grid_cost_funtion" => Dict(:gl=>(10, 60), :ka=>(0.5, 10),
                                             :npoints => (100, 100)),
                 "solve_grid" => Dict(:gl=>(10, 60), :ka=>(0.5, 10),
                                      :npoints => (5, 5)))
        push!(options_dicts, d)
    end
end

for ampl in [0.05, 0.2]
    for σv in [0, 0.03]
        d = Dict("data_generator" => Dict(:t=>"inverted_pendulum",
                                          :N=>1000, :ampl=>ampl, :rep=>20,
                                          :σv=>σv, :seed=>1),
                 "pem" => Dict(:model=>"output_error"),
                 "grid_cost_funtion" => Dict(:gl=>(10, 60), :ka=>(0.5, 10),
                                             :npoints => (100, 100)),
                 "solve_grid" => Dict(:gl=>(10, 60), :ka=>(0.5, 10),
                                      :npoints => (5, 5)))
        push!(options_dicts, d)
    end
end


shuffle!(options_dicts)
map(run_and_save, options_dicts)
