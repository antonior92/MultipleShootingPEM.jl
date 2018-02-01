mutable struct OptimizationProblem{N, Ny, Nx, Nθ, M}
    ms::MultipleShooting{Vector{Float64}, N, Ny, Nx, Nθ, M}
    # Parameter vector
    θ_ext::Vector{Float64}
    θ_aux::Vector{Float64}
    x0_list_aux::Vector{Vector{Float64}}
    # Gradient
    grad::Vector{Float64}
    gradθ_aux::Vector{Float64}
    gradx0_aux::Vector{Vector{Float64}}
    # Hessian
    hessp::Vector{Float64}
    hesspθ_aux::Vector{Float64}
    hesspx0_aux::Vector{Vector{Float64}}
    # Constraint
    constr::Vector{Float64}
end

function OptimizationProblem(f::Function, g::Function,
                             x0_list::Vector{Vector{Float64}},
                             y::Vector{Vector{Float64}},
                             k0_list::Vector{Int}, θ,
                             list_procs::Vector{Int},
                             loss=L2DistLoss())
   ms = MultipleShooting(f, g, x0_list, y,
                         k0_list, θ, list_procs, loss)
   # Problem dimensions
   M = length(x0_list)
   N = length(y)
   Ny = length(y[1])
   Nx = length(x0_list[1])
   Nθ = length(θ)
   # Parameter Vector
   θ_ext = zeros(Nθ + M*Nx)
   θ_aux = zeros(Nθ)
   x0_list_aux = Vector{Vector{Float64}}(M)
   for i = 1:M
       x0_list_aux[i] = zeros(Nx)
   end
   # Gradient
   grad = zeros(Nθ + M*Nx)
   gradθ_aux = zeros(Nθ)
   gradx0_aux = Vector{Vector{Float64}}(M)
   for i = 1:M
       gradx0_aux[i] = zeros(Nx)
   end
   # Hessian
   hessp = zeros(Nθ + M*Nx)
   hesspθ_aux = zeros(Nθ)
   hesspx0_aux = Vector{Vector{Float64}}(M)
   for i = 1:M
       hesspx0_aux[i] = zeros(Nx)
   end
   # Constraint
   constr = zeros((M-1)*Nx)
   # Call constructor
   OptimizationProblem{N, Ny, Nx, Nθ, M}(
      ms, θ_ext, θ_aux, x0_list_aux, grad, gradθ_aux, gradx0_aux,
      hessp, hesspθ_aux, hesspx0_aux, constr)
end

function cost_function{N, Ny, Nx, Nθ, M}(
        opt::OptimizationProblem{N, Ny, Nx, Nθ, M},
        θ_ext::Vector{Float64})
    update_simulation!(opt, θ_ext)
    return cost_function(opt.ms)
end

function gradient{N, Ny, Nx, Nθ, M}(
        opt::OptimizationProblem{N, Ny, Nx, Nθ, M},
        θ_ext::Vector{Float64})
    update_simulation!(opt, θ_ext)
    gradient!(opt.gradθ_aux, opt.ms, "θ")
    gradient!(opt.gradx0_aux, opt.ms, "x0")
    return build_extended_vector!(opt.grad, opt.gradθ_aux, opt.gradx0_aux,
                                  Nθ, M, Nx)
end

function hessian{N, Ny, Nx, Nθ, M}(
        opt::OptimizationProblem{N, Ny, Nx, Nθ, M},
        θ_ext::Vector{Float64})
    update_simulation!(opt, θ_ext)
    function hessp(p)
        hessian_approx!(opt.hesspθ_aux, opt.ms, "θ")
        hessian_approx!(opt.hesspx0_aux, opt.ms, "x0")
        return build_extended_vector!(opt.hessp, opt.hesspθ, opt.hesspx0,
                                     Nθ, M, Nx)
    end
    return scipy_sps[:LinearOperator]((Nθ+Nx*M, Nθ+Nx*M), matvec=hessp)
end

function constraint{N, Ny, Nx, Nθ, M}(
        opt::OptimizationProblem{N, Ny, Nx, Nθ, M},
        θ_extended::Vector{Float64})
    update_simulation!(opt, θ_ext)
    return constr!(opt.constr, opt.ms)
end


function update_simulation!{N, Ny, Nx, Nθ, M}(
        opt::OptimizationProblem{N, Ny, Nx, Nθ, M}, θ_ext::Vector{Float64})
    if θ_ext != opt.θ_ext
        copy!(opt.θ_ext, θ_ext)
        read_extended_vector!(opt.θ_aux, opt.x0_list_aux, opt.θ_ext, Nθ, M, Nx)
        copy!(opt.θ_ext, θ_ext)
        new_simulation!(opt.ms, opt.x0_list_aux, opt.θ_aux)
        return true
    else
        return false
    end
end

function build_extended_vector!(θ_extended, θ, x0_list, Nθ, M, Nx)
    k = 1
    for i = 1:Nθ
        θ_extended[k] = θ[i]
        k += 1
    end
    for i = 1:M
        for j = 1:Nx
            θ_extended[k] = x0_list[i][j]
            k += 1
        end
    end
    return θ_extended
end

function read_extended_vector!(θ, x0_list, θ_extended, Nθ, M, Nx)
    k = 1
    for i = 1:Nθ
        θ[i] = θ_extended[k]
        k += 1
    end
    for i = 1:M
        for j = 1:Nx
            x0_list[i][j] = θ_extended[k]
            k += 1
        end
    end
    return θ, x0_list
end
