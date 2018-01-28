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

function gradient_θ!{T, N, Ny, Nx, Nθ, M}(
        grad::Vector{Float64},
        grad_remote::Vector{Union{Future, Vector{Float64}}},
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M},
        loss=L2DistLoss(), accumulat=false)
    variable="θ"
    # Set initial values to zero if not accumulating
    nprocess = maximum(ms.list_procs)
    for proc in 1:nprocess
        if proc == 1
            fill!(grad_remote[proc], 0)
        else
            grad_remote[proc] = remotecall(fill!, proc,
                                           fetch(grad_remote[proc]), 0)
        end
    end
    if !accumulat
        fill!(grad, 0)
    end
    # Make all the computations on remote instances
    first_evaluation = trues(nprocess)
    for i = 1:M
        proc = ms.list_procs[i]
        if first_evaluation[proc]
            first_evaluation[proc] = false
        end
        if proc == 1
            gradient!(grad_remote[proc], ms.simulations[i],
                      variable, loss, !first_evaluation[proc])
        else
            grad_remote[proc] = remotecall(gradient!, proc,
                                    fetch(grad_remote[proc]),
                                    fetch(ms.simulations[i]), variable, loss,
                                    !first_evaluation[proc])
        end
    end
    # Put everything togeter
    for proc = 1:nprocess
        if proc == 1
            Base.LinAlg.axpy!(1, grad_remote[proc], grad)
        else
            Base.LinAlg.axpy!(1, fetch(grad_remote[proc]), grad)
        end
    end
end

function initialize_instance_everywhere{T}(instance::T, nprocess)
    instance_remote = Vector{Union{Future, T}}(nprocess)
    instance_remote[1] = deepcopy(instance)
    for p in 2:nprocess
            instance_remote[p] = remotecall(deepcopy, p, instance)
    end
    return instance_remote
end
