addprocs(2)
@everywhere import MultipleShootingPEM
@everywhere include("ex3_base.jl")
@everywhere import Example3
@everywhere using JLD2
@everywhere using JSON

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

#sim_len = [2^i for i in 0:10]
sim_len = [32, 1024]
options_dicts = []
# Near stable region
d = Dict("data_generator" => Dict(:t=>"pendulum", :N=>N,
                                  :σw=>0., :σv=>0.03,
                                  :ampl=>10, :rep=>16,
                                  :seed=>1),
         "pem" => Dict(:model=>"output_error"),
         "grid_cost_funtion" => Dict(:gl=>(20, 50), :ka=>(0.5, 6),
                                     :npoints => (100, 100)),
         "solve_grid" => Dict(:gl=>(20, 50), :ka=>(0.5, 6),
                              :npoints => (5, 5)),
         "sim_len" => sim_len)
push!(options_dicts, d)
# Full turns around center
d = Dict("data_generator" => Dict(:t=>"pendulum", :N=>N,
                                  :σw=>0., :σv=>0.03,
                                  :ampl=>50, :rep=>16,
                                  :seed=>1),
         "pem" => Dict(:model=>"output_error"),
         "grid_cost_funtion" => Dict(:gl=>(20, 50), :ka=>(0.5, 6),
                                     :npoints => (100, 100)),
         "solve_grid" => Dict(:gl=>(20, 50), :ka=>(0.5, 6),
                              :npoints => (5, 5)),
         "sim_len" => sim_len)
push!(options_dicts, d)
# Near unstable region
d = Dict("data_generator" => Dict(:t=>"inverted_pendulum",
                                  :N=>N, :ampl=>0.2, :rep=>16,
                                  :σv=>0, :seed=>1),
         "pem" => Dict(:model=>"output_error"),
         "grid_cost_funtion" => Dict(:gl=>(20, 50), :ka=>(0.5, 6),
                                     :npoints => (100, 100)),
         "solve_grid" => Dict(:gl=>(20, 50), :ka=>(0.5, 6),
                              :npoints => (5, 5)),
         "sim_len" => sim_len)
push!(options_dicts, d)

pmap(run_and_save, options_dicts)
