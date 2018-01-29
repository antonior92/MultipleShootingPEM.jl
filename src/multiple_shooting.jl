struct MultipleShooting{T, N, Ny, Nx, Nθ, M}
    simulations
    list_procs
    list_N
end

function MultipleShooting{T}(f::Function, g::Function,
                             x0_list::Vector{T}, y::Vector{T},
                             k0_list::Vector{Int}, θ,
                             list_procs::Vector{Int})
    # Order list of procs
    sort!(list_procs)
    # Get lengths
    M = length(x0_list)
    N = length(y)
    Ny = length(y[1])
    Nx = length(x0_list[1])
    Nθ = length(θ)
    # Simulate
    k0 = k0_list[1]
    grid = [k0_list; k0+N]
    list_N = Vector{Int}(M)
    simulations = Vector{Union{Future,OneShootSimulation}}(M)
    for i =1:M
        proc = list_procs[i]; x0 = x0_list[i]; k0 = k0_list[i];
        yi = y[grid[i]:grid[i+1]-1]; list_N[i] = length(yi);
        if proc == 1
            simulations[i] = OneShootSimulation(f, g, x0, yi, k0, θ)
        else
            simulations[i] = @spawnat(proc, OneShootSimulation(f, g, x0,
                                                               yi, k0, θ))
        end
    end
    MultipleShooting{T, N, Ny, Nx, Nθ, M}(simulations, sort(list_procs), list_N)
end

function new_simulation!{T, N, Ny, Nx, Nθ, M}(
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M}, x0_list::Vector{T}, θ)
    for i = 1:M
        proc = ms.list_procs[i]; x0 = x0_list[i]
        if proc == 1
            ms.simulations[i] = new_simulation!(ms.simulations[i], x0, θ)
        else
            ms.simulations[i] = remotecall(new_simulation!, proc,
                fetch(ms.simulations[i]), x0, θ)
        end
    end
    return ms
end

function cost_function{T, N, Ny, Nx, Nθ, M}(
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M},
        loss=L2DistLoss())
    cost = 0
    for i = 1:M
        proc = ms.list_procs[i]
        if proc == 1
            cost += cost_function(ms.simulations[i], loss)
        else
            cost += remotecall_fetch(cost_function, proc,
                                     fetch(ms.simulations[i]), loss)
        end
    end
    return cost
end

function derivatives_θ!{T, N, Ny, Nx, Nθ, M}(
        dvec::Vector{Float64},
        dvec_remote::Vector{Union{Future, Vector{Float64}}},
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M},
        loss=L2DistLoss(), accumulat=false, dtype="gradient",
        p::Vector{Float64}=Float64[])
    variable="θ"
    # Set initial values to zero if not accumulating
    nprocess = maximum(ms.list_procs)
    for proc in 1:nprocess
        if proc == 1
            fill!(dvec_remote[proc], 0)
        else
            dvec_remote[proc] = remotecall(fill!, proc,
                                           fetch(dvec_remote[proc]), 0)
        end
    end
    if !accumulat
        fill!(dvec, 0.0)
    end
    # Make all the computations on remote instances
    first_evaluation = trues(nprocess)
    for i = 1:M
        proc = ms.list_procs[i]
        if first_evaluation[proc]
            first_evaluation[proc] = false
        end
        if proc == 1
            if dtype == "gradient"
                gradient!(dvec_remote[proc], ms.simulations[i],
                          variable, loss, !first_evaluation[proc])
            elseif dtype == "hessian_approx"
                hessian_aprox!(dvec_remote[proc], ms.simulations[i], p,
                               variable, loss, !first_evaluation[proc])
            end
        else
            if dtype == "gradient"
                dvec_remote[proc] = remotecall(
                    gradient!, proc, fetch(dvec_remote[proc]),
                    fetch(ms.simulations[i]), variable, loss,
                    !first_evaluation[proc])
            elseif dtype == "hessian_approx"
                dvec_remote[proc] = remotecall(
                    hessian_approx!, proc, fetch(dvec_remote[proc]),
                    fetch(ms.simulations[i]), p, variable, loss,
                    !first_evaluation[proc])
            end
        end
    end
    # Put everything togeter
    for proc = 1:nprocess
        if proc == 1
            Base.LinAlg.axpy!(1, dvec_remote[proc], dvec)
        else
            Base.LinAlg.axpy!(1, fetch(dvec_remote[proc]), dvec)
        end
    end
end


function derivatives_x0!{T, N, Ny, Nx, Nθ, M}(
        dvec::Vector{Vector{Float64}},
        dvec_remote::Vector{Union{Future, Vector{Float64}}},
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M},
        loss=L2DistLoss(), accumulat=false, dtype="gradient",
        p::Vector{Float64}=Float64[])
    variable="x0"
    # Set initial values to zero if not accumulating
    if !accumulat
        for i = 1:M
            fill!(dvec[i], 0.0)
        end
    end
    # Make all the computations on remote instances
    for i = 1:M
        proc = ms.list_procs[i]
        if proc == 1
            if dtype == "gradient"
                gradient!(dvec_remote[i], ms.simulations[i],
                          variable, loss, false)
            elseif dtype == "hessian_approx"
                hessian_approx!(dvec_remote[i], ms.simulations[i],
                                p, variable, loss, false)
            end
        else
            if dtype == "gradient"
                dvec_remote[i] = remotecall(
                    gradient!, proc, fetch(dvec_remote[i]),
                    fetch(ms.simulations[i]), variable,
                    loss, false)
            elseif dtype == "hessian_approx"
                dvec_remote[i] = remotecall(
                    hessian_approx!, proc, fetch(dvec_remote[i]),
                    fetch(ms.simulations[i]), p, variable,
                    loss, false)
            end
        end
    end
    # Put everything togeter
    for i = 1:M
        proc = ms.list_procs[i]
        if proc == 1
            Base.LinAlg.axpy!(1, dvec_remote[i], dvec[i])
        else
            Base.LinAlg.axpy!(1, fetch(dvec_remote[i]), dvec[i])
        end
    end
end

function deepcopy_everywhere{T}(instance::T, list_procs)
    if all(list_procs .== 1)
        instance_remote = Vector{Union{T}}(length(list_procs))
    else
        instance_remote = Vector{Union{Future, T}}(length(list_procs))
    end
    for i in 1:length(list_procs)
        proc = list_procs[i]
        if proc == 1
            instance_remote[i] = deepcopy(instance)
        else
            instance_remote[i] = remotecall(deepcopy, proc, instance)
        end
    end
    return instance_remote
end

function gradient!{T, N, Ny, Nx, Nθ, M}(
        grad,
        grad_remote,
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M},
        variable="θ", loss=L2DistLoss(), accumulat=false)
    if variable == "θ"
        derivatives_θ!(grad, grad_remote, ms, loss, accumulat, "gradient")
    elseif variable == "x0"
        derivatives_x0!(grad, grad_remote, ms, loss, accumulat, "gradient")
    end
end


function hessian_approx!{T, N, Ny, Nx, Nθ, M}(
        hessp,
        hessp_remote,
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M}, p,
        variable="θ", loss=L2DistLoss(), accumulat=false)
    if variable == "θ"
        derivatives_θ!(hessp, hessp_remote, ms, loss,
                       accumulat, "hessian_approx", p)
    elseif variable == "x0"
        derivatives_x0!(hessp, hessp_remote, ms,
                        loss, accumulat, "hessian_approx", p)
    end
end
