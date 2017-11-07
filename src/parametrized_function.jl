# Parametrized function. Instancialize a function that has a
# known dependency on parameter θ.
struct ParametrizedFunction <: Function
    f::Function
    θ
end

# Possible callers
(pf::ParametrizedFunction)(y, x, args...) = pf.f(y, x, pf.θ, args...)

# Auxiliar function for matrix multiplication
function gaxpy!(A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64},
                C::AbstractMatrix{Float64})
    n, p = size(A)
    p, m = size(B)
    for j = 1:m
        for i = 1:n
            for k = 1:p
                C[i, j] += A[i, k]*B[k, j]
            end
        end
    end
end

# Be a function: y = f(x, args...) that depends uppon
# a parameter vector θ. We expect as argument a function that
# is able to evaluate f and to compute its derivatives df/dθ, df/dx.
# The expected signature is:
# Jf((y, df/dθ, df/dx), x, args...)
function sensitivity_equation(Jf::Function, jacobian_buffer)
    function f(y_extended, x_extended, args...; J=jacobian_buffer)
        # Get points
        x, dx = x_extended
        y, dy = y_extended
        # Evaluate and compute derivatives
        Jf((y, dy, J), x, args...)
        # Using the chain rule:
        # dy/dθ += dy/dx*dx/dθ
        gaxpy!(J, dx, dy)
    end
    return f
end

function sensitivity_equation(Jf::ParametrizedFunction, jacobian_buffer)
    f = sensitivity_equation(Jf.f, jacobian_buffer)
    return ParametrizedFunction(f, Jf.θ)
end
