struct MultipleShooting{T, N, Ny, Nx, Nθ, M}
    simulations
    list_N
end

function MultipleShooting{T}(f::Function, g::Function,
                             x0_list::Vector{T}, y::Vector{T},
                             k0_list::Vector{Int}, θ,
                             loss=L2DistLoss())
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
        x0 = x0_list[i]; k0 = k0_list[i];
        yi = y[grid[i]:grid[i+1]-1]; list_N[i] = length(yi);
        simulations[i] = OneShootSimulation(f, g, x0, yi, k0, θ, loss)

    end
    MultipleShooting{T, N, Ny, Nx, Nθ, M}(simulations, list_N)
end

function new_simulation!{T, N, Ny, Nx, Nθ, M}(
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M}, x0_list::Vector{T}, θ)
    for i = 1:M
        x0 = x0_list[i]
        ms.simulations[i] = new_simulation!(ms.simulations[i], x0, θ)
    end
    return ms
end

function cost_function{T, N, Ny, Nx, Nθ, M}(
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M})
    cost = 0
    for i = 1:M
        cost += cost_function(ms.simulations[i])
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
    if !accumulat
        fill!(dvec, 0.0)
    end
    # Make all the computations on remote instances
    for i = 1:M
        if dtype == "gradient"
            gradient!(dvec, ms.simulations[i],
                      variable, true)
        elseif dtype == "hessian_approx"
            hessian_approx!(dvec, ms.simulations[i], p,
                           variable, true)
        end
    end
    return dvec
end


function derivatives_x0!{T, N, Ny, Nx, Nθ, M}(
        dvec::Vector{Vector{Float64}},
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M},
        accumulat=false, dtype="gradient",
        p::Vector{Vector{Float64}}=Vector{Float64}[])
    variable="x0"
    # Set initial values to zero if not accumulating
    if !accumulat
        for i = 1:M
            fill!(dvec[i], 0.0)
        end
    end
    # Make all the computations on remote instances
    for i = 1:M
        if dtype == "gradient"
            gradient!(dvec[i], ms.simulations[i],
                      variable, false)
        elseif dtype == "hessian_approx"
            hessian_approx!(dvec[i], ms.simulations[i],
                            p[i], variable, false)
        end
    end
    return dvec
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
        copy!(c[i], ms.simulations[i].x)
    end
    # subtract x0
    for i = 2:M
        Base.LinAlg.axpy!(-1, ms.simulations[i].x0, c[i-1])
    end
    return c
end

function constr_jac!{T, N, Ny, Nx, Nθ, M}(
        jac::Vector{Matrix{Float64}},
        ms::MultipleShooting{T, N, Ny, Nx, Nθ, M},
        variable="θ")
    # Get x
    for i = 1:M-1
        if variable == "x0"
            jac[i] .= get_dxdx0(ms.simulations[i])
        elseif variable == "θ"
            jac[i] .= get_dxdθ(ms.simulations[i])
        end
    end
    return jac
end
