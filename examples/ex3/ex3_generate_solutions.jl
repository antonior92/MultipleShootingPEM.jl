addprocs(7)
@everywhere import MultipleShootingPEM
@everywhere include("ex3_base.jl")
@everywhere import Example3
@everywhere using JLD2
@everywhere using Plots
@everywhere using JSON
@everywhere pyplot()

@everywhere N=1024
@everywhere function run_and_save(options_dict)
    println("Create directory")
    if ~isdir("solutions")
        mkdir("solutions")
    end
    time = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS-sss")
    root_dir = "solutions/sol_"*time*"_"*string(rand(1:100000))
    mkdir(root_dir)
    for sim_len in options_dict["sim_len"]
        println("")
        println("**************************")
        println("sim_len = "*string(sim_len))
        options_dict["pem"] = merge(options_dict["pem"], Dict(:sim_len=>sim_len))
        results = Example3.runner(options_dict)
        println("Save results...")
        dir = root_dir*"/sim_len_"*string(sim_len)
        mkdir(dir)
        # Save all data
        JLD2.@save dir*"/results.jld2" results
        # Save config file separately to be easy to check it
        json_string = JSON.json(results["options"])
        open(dir*"/config.json","w") do f
            write(f, json_string)
        end
    end
end

sim_len = [2^i for i in 0:10]
options_dicts = []
for ampl in [10, 50]
    for σv in [0, 0.2]
        d = Dict("data_generator" => Dict(:t=>"pendulum", :N=>N,
                                          :σw=>0., :σv=>σv,
                                          :ampl=>ampl, :rep=>20,
                                          :seed=>1),
                 "pem" => Dict(:model=>"output_error"),
                 "grid_cost_funtion" => Dict(:gl=>(10, 60), :ka=>(0.5, 10),
                                             :npoints => (100, 100)),
                 "solve_grid" => Dict(:gl=>(10, 60), :ka=>(0.5, 10),
                                      :npoints => (5, 5)),
                 "sim_len" => sim_len)
        push!(options_dicts, d)
    end
end

for ampl in [0.05, 0.2]
    for σv in [0, 0.03]
        d = Dict("data_generator" => Dict(:t=>"inverted_pendulum",
                                          :N=>1000, :ampl=>ampl, :rep=>20,
                                          :σv=>σv, :seed=>1),
                 "pem" => Dict(:model=>"output_error"),
                 "grid_cost_funtion" => Dict(:gl=>(0, 100), :ka=>(0, 20),
                                             :npoints => (100, 100)),
                 "solve_grid" => Dict(:gl=>(0, 60), :ka=>(0.5, 10),
                                      :npoints => (5, 5)),
                 "sim_len" => sim_len)
        push!(options_dicts, d)
    end
end


shuffle!(options_dicts)
pmap(run_and_save, options_dicts)
