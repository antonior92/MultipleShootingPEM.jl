# Parametrized function. Instancialize a function that has a
# known dependency on parameter θ.
struct ParametrizedFunction <: Function
    f::Function
    θ
end

(pf::ParametrizedFunction)(y, x, args...) = pf.f(y, x, pf.θ, args...)

# Be a function: y = f(x, args...) that depends uppon
# a parameter vector θ, for x dependent on θ and ϕ.
# Provides a routine for computing y, dy/dθ and dy/dϕ,
# given x, dx/dθ and dx/dϕ.
# We expect as argument a function:
# Jf((y, df/dθ, df/dx), x, args...)
# that is able to evaluate the function output y and to
# compute its derivatives df/dθ, df/dx.
function sensitivity_equation(Jf::Function, jacobian_buffer)
    function f(y_extended, x_extended, args...; dydx=jacobian_buffer)
        # Expand
        x, dxdθ, dxdϕ = x_extended
        y, dydθ, dydϕ = y_extended
        # Evaluate and compute derivatives
        Jf((y, dydx, dydθ), x, args...)
        # Using the chain rule:
        # dy/dθ += dy/dx*dx/dθ
        Base.LinAlg.BLAS.gemm!('N', 'N', 1.0, dydx, dxdθ, 1.0, dydθ)
        # dy/dϕ = dy/dx*dx/dxϕ
        A_mul_B!(dydϕ, dydx, dxdϕ)
    end
    return f
end

function sensitivity_equation(Jf::ParametrizedFunction, jacobian_buffer)
    f = sensitivity_equation(Jf.f, jacobian_buffer)
    return ParametrizedFunction(f, Jf.θ)
end
