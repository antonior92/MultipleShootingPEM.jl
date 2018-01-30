struct MultipleShooting{T, N, Ny, Nx, Nθ, M}
    simulations
    list_procs
    list_N
    dvecθ_remote
    dvecx0_remote
end

function MultipleShooting{T}(f::Function, g::Function,
                             x0_list::Vector{T}, y::Vector{T},
                             k0_list::Vector{Int}, θ,
                             list_procs::Vector{Int},
                             loss=L2DistLoss())
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
            simulations[i] = OneShootSimulation(f, g, x0, yi, k0, θ, loss)
        else
            simulations[i] = @spawnat(proc,
                OneShootSimulation(f, g, x0, yi, k0, θ, loss))
        end
    end
    dvecθ_remote = deepcopy_everywhere(zeros(Nθ), unique(list_procs))
    dvecx0_remote = deepcopy_everywhere(zeros(Nx), list_procs)
    MultipleShooting{T, N, Ny, Nx, Nθ, M}(simulations,
        sort(list_procs), list_N, dvecθ_remote, dvecx0_remote)
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
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M})
    cost = 0
    for i = 1:M
        proc = ms.list_procs[i]
        if proc == 1
            cost += cost_function(ms.simulations[i])
        else
            cost += remotecall_fetch(cost_function, proc,
                                     fetch(ms.simulations[i]))
        end
    end
    return cost
end

function derivatives_θ!{T, N, Ny, Nx, Nθ, M}(
        dvec::Vector{Float64},
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M},
        accumulat=false, dtype="gradient",
        p::Vector{Float64}=Float64[])
    variable="θ"
    # Set initial values to zero if not accumulating
    list_workers = unique(ms.list_procs)
    nprocess = length(list_workers)
    for ind = 1:nprocess
        proc = list_workers[ind]
        if proc == 1
            fill!(ms.dvecθ_remote[ind], 0)
        else
            ms.dvecθ_remote[ind] = remotecall(fill!, proc,
                                        fetch(ms.dvecθ_remote[ind]), 0)
        end
    end
    if !accumulat
        fill!(dvec, 0.0)
    end
    # Make all the computations on remote instances
    first_evaluation = trues(nprocess)
    for i = 1:M
        proc = ms.list_procs[i]
        ind = findfirst(proc .== list_workers)
        if all(first_evaluation[ind])
            first_evaluation[ind] = false
        end
        if proc == 1
            if dtype == "gradient"
                gradient!(ms.dvecθ_remote[proc], ms.simulations[i],
                          variable, !first_evaluation[proc])
            elseif dtype == "hessian_approx"
                hessian_aprox!(ms.dvecθ_remote[proc], ms.simulations[i], p,
                               variable, !first_evaluation[proc])
            end
        else
            if dtype == "gradient"
                ms.dvecθ_remote[ind] = remotecall(
                    gradient!, proc, fetch(ms.dvecθ_remote[ind]),
                    fetch(ms.simulations[i]), variable,
                    !first_evaluation[ind])
            elseif dtype == "hessian_approx"
                ms.dvecθ_remote[ind] = remotecall(
                    hessian_approx!, proc, fetch(ms.dvecθ_remote[ind]),
                    fetch(ms.simulations[i]), p, variable,
                    !first_evaluation[ind])
            end
        end
    end
    # Put everything togeter
    for ind = 1:nprocess
        if list_workers[ind] == 1
            Base.LinAlg.axpy!(1, ms.dvecθ_remote[ind], dvec)
        else
            Base.LinAlg.axpy!(1, fetch(ms.dvecθ_remote[ind]), dvec)
        end
    end
    return dvec
end


function derivatives_x0!{T, N, Ny, Nx, Nθ, M}(
        dvec::Vector{Vector{Float64}},
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M},
        accumulat=false, dtype="gradient",
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
                gradient!(ms.dvecx0_remote[i], ms.simulations[i],
                          variable, false)
            elseif dtype == "hessian_approx"
                hessian_approx!(ms.dvecx0_remote[i], ms.simulations[i],
                                p, variable, false)
            end
        else
            if dtype == "gradient"
                ms.dvecx0_remote[i] = remotecall(
                    gradient!, proc, fetch(ms.dvecx0_remote[i]),
                    fetch(ms.simulations[i]), variable,
                    false)
            elseif dtype == "hessian_approx"
                ms.dvecx0_remote[i] = remotecall(
                    hessian_approx!, proc, fetch(ms.dvecx0_remote[i]),
                    fetch(ms.simulations[i]), p, variable,
                    false)
            end
        end
    end
    # Put everything togeter
    for i = 1:M
        proc = ms.list_procs[i]
        if proc == 1
            Base.LinAlg.axpy!(1, ms.dvecx0_remote[i], dvec[i])
        else
            Base.LinAlg.axpy!(1, fetch(ms.dvecx0_remote[i]), dvec[i])
        end
    end
    return dvec
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
        grad, ms::MultipleShooting{T, N, Ny, Nx, Nθ, M},
        variable="θ", accumulat=false)
    if variable == "θ"
        derivatives_θ!(grad, ms, accumulat, "gradient")
    elseif variable == "x0"
        derivatives_x0!(grad, ms, accumulat, "gradient")
    end
end


function hessian_approx!{T, N, Ny, Nx, Nθ, M}(
        hessp, ms::MultipleShooting{T, N, Ny, Nx, Nθ, M}, p,
        variable="θ", accumulat=false)
    if variable == "θ"
        derivatives_θ!(hessp, ms, accumulat, "hessian_approx", p)
    elseif variable == "x0"
        derivatives_x0!(hessp, ms, accumulat, "hessian_approx", p)
    end
end

function constr!{T, N, Ny, Nx, Nθ, M}(
        c::Vector{Vector{Float64}},
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M})
    # Get x
    for i = 1:M-1
        proc = ms.list_procs[i]
        if proc == 1
            copy!(c[i], ms.simulations[i].x)
        else
            copy!(c[i], remotecall_fetch(get_x, proc, fetch(ms.simulations[i])))
        end
    end
    # subtract x0
    for i = 2:M
        proc = ms.list_procs[i]
        if proc == 1
            Base.LinAlg.axpy!(-1, ms.simulations[i].x0, c[i-1])
        else
            Base.LinAlg.axpy!(-1,
                remotecall_fetch(get_x0, proc, fetch(ms.simulations[i])),
                c[i-1])
        end
    end
    return c
end

function constr_jac!{T, N, Ny, Nx, Nθ, M}(
        jac::Vector{Matrix{Float64}},
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M},
        variable="θ")
    # Get x
    for i = 1:M-1
        proc = ms.list_procs[i]
        if proc == 1
            if variable == "x0"
                jac[i] .= get_dxdx0(ms.simulations[i])
            elseif variable == "θ"
                jac[i] .= get_dxdθ(ms.simulations[i])
            end
        else
            if variable == "x0"
                jac[i] .= remotecall_fetch(get_dxdx0, proc,
                                           fetch(ms.simulations[i]))
            elseif variable == "θ"
                jac[i] .= remotecall_fetch(get_dxdθ, proc,
                                           fetch(ms.simulations[i]))
            end
        end
    end
    return jac
end
