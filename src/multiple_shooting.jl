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
    MultipleShooting{T, N, Ny, Nx, Nθ, M}(simulations, list_procs, list_N)
end

function new_simulation!{T, N, Ny, Nx, Nθ, M}(
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M}, x0_list::Vector{T}, θ)
    for i = 1:M
        proc = ms.list_procs[i]; x0 = x0_list[i]
        if proc == 1
            ms.simulations[i] = new_simulation!(ms.simulations[i], x0, θ)
        else
            ms.simulations[i] = @spawnat(proc,
                new_simulation!(fetch(ms.simulations[i]), x0, θ))
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
