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
    # Multiplicative vector p
    p_θ::Vector{Float64}
    p_x0::Vector{Vector{Float64}}
    # Hessian
    hessp::Vector{Float64}
    hesspθ_aux::Vector{Float64}
    hesspx0_aux::Vector{Vector{Float64}}
    # Constraint
    constr::Vector{Vector{Float64}}
    # Jacobian
    jacobian::SparseMatrixCOO{Float64,Int64}
    jacobian_θ::Vector{Matrix{Float64}}
    jacobian_x0::Vector{Matrix{Float64}}
end

function OptimizationProblem(f::Function, g::Function,
                             x0_list::Vector{Vector{Float64}},
                             y::Vector{Vector{Float64}},
                             k0_list::Vector{Int}, θ,
                             list_procs::Vector{Int};
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
   # Multiplicative_vector_p
   p_θ = zeros(Nθ)
   p_x0 = Vector{Vector{Float64}}(M)
   for i = 1:M
       p_x0[i] = zeros(Nx)
   end
   # Hessian
   hessp = zeros(Nθ + M*Nx)
   hesspθ_aux = zeros(Nθ)
   hesspx0_aux = Vector{Vector{Float64}}(M)
   for i = 1:M
       hesspx0_aux[i] = zeros(Nx)
   end
   # Constraint
   constr = Vector{Vector{Float64}}(M-1)
   for i = 1:M-1
       constr[i] = zeros(Nx)
   end
   # Jacobian
   jacobian_θ = Vector{Matrix{Float64}}(M-1)
   for i = 1:M-1
       jacobian_θ[i] = ones(Nx, Nθ)
   end
   jacobian_x0 = Vector{Matrix{Float64}}(M-1)
   for i = 1:M-1
       jacobian_x0[i] = ones(Nx, Nx)
   end
   i, j = jacobian_indices(Nθ, M, Nx)
   nnz = length(i)
   v = Vector{Flaot64}(nnz)
   jacobian = SparseMatrixCOO{Float64,Int64}(Nx*(M-1), Nθ + M*Nx, i, j, v)
   jacobian_from_matrices!(jacobian, jacobian_θ, jacobian_x0,
                           Nθ, M, Nx; reset_constants=true)
   # Call constructor
   OptimizationProblem{N, Ny, Nx, Nθ, M}(
      ms, θ_ext, θ_aux, x0_list_aux, grad, gradθ_aux, gradx0_aux, p_θ, p_x0,
      hessp, hesspθ_aux, hesspx0_aux, constr, jacobian, jacobian_θ, jacobian_x0)
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
        read_extended_vector!(opt.p_θ, opt.p_x0, p, Nθ, M, Nx)
        hessian_approx!(opt.hesspθ_aux, opt.ms, opt.p_θ, "θ")
        hessian_approx!(opt.hesspx0_aux, opt.ms, opt.p_x0, "x0")
        return build_extended_vector!(opt.hessp, opt.hesspθ_aux, opt.hesspx0_aux,
                                      Nθ, M, Nx)
    end
    return scipy_sps["LinearOperator"]((Nθ+M*Nx, Nθ+M*Nx), matvec=hessp)
end

function constraint{N, Ny, Nx, Nθ, M}(
        opt::OptimizationProblem{N, Ny, Nx, Nθ, M},
        θ_ext::Vector{Float64})
    update_simulation!(opt, θ_ext)
    return constr!(opt.constr, opt.ms)
end


function jacobian{N, Ny, Nx, Nθ, M}(
        opt::OptimizationProblem{N, Ny, Nx, Nθ, M},
        θ_ext::Vector{Float64})
    update_simulation!(opt, θ_ext)
    constr_jac!(opt.jacobian_θ, opt.ms, "θ")
    constr_jac!(opt.jacobian_x0, opt.ms, "x0")
    return jacobian_from_matrices!(opt.jacobian, opt.jacobian_θ, opt.jacobian_x0,
                                   Nθ, M, Nx; reset_constants=false)
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

function jacobian_indices(Nθ, M, Nx)
    # Number of nonzero elements
    nnz = (M-1)*Nx*(Nθ+Nx+1)
    # Initialize indeces vectors
    i = Vector{Int64}(nnz)
    j = Vector{Int64}(nnz)
    # Counter
    k = 1
    # Include elements from ``jacobian_θ`` matrices
    for rows_offset = 0:Nx:(M-1)*Nx
        for rows = 1:Nx
            for cols = 1:Nθ
                i[k] = rows + rows_offset
                j[k] = cols
                k += 1
            end
        end
    end
    # Include elements from ``jacobian_x0`` matrices
    cols_offset = Nθ
    for offset = 0:Nx:(M-1)*Nx
        for rows = 1:Nx
            for cols = 1:Nx
                i[k] = rows + offset
                j[k] = cols + offset + cols_offset
                k += 1
            end
        end
    end
    # Include indices relative to -eye(Nx) matrices
    cols_offset = Nθ
    for offset = Nx:Nx:M*Nx
        for rows = 1:Nx
            i[k] = rows + offset
            j[k] = rows + offset + cols_offset
            k += 1
        end
    end
    return i, j
end


function jacobian_from_matrices!(jacobian::SparseMatrixCOO,
                                 jacobian_θ, jacobian_x0, Nθ, M, Nx;
                                 reset_constants=false)
    # Counter
    k = 1
    # Include elements from ``jacobian_θ`` matrices
    for matrix_number = 1:(M-1)
        for rows = 1:Nx
            for cols = 1:Nθ
                jacobian.v[k] = jacobian_θ[matrix_number][rows, cols]
                k += 1
            end
        end
    end
    # Include elements from ``jacobian_x0`` matrices
    for matrix_number = 1:(M-1)
        for rows = 1:Nx
            for cols = 1:Nx
                jacobian.v[k] = jacobian_x0[matrix_number][rows, cols]
                k += 1
            end
        end
    end
    # Include indices relative to -eye(Nx) matrices
    if reset_constants
        for matrix_number = 1:(M-1)
            for rows = 1:Nx
                jacobian.v[k] = -1
                k += 1
            end
        end
    end
    return jacobian
end
