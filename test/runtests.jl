import MultipleShootingPEM
ms = MultipleShootingPEM
using Base.Test
using Calculus

include("test_util.jl")
include("test_sensitivity_equations.jl")
include("test_space_state_model.jl")
include("test_one_shoot_simulation.jl")
include("test_multiple_shooting.jl")
include("test_optimization_problem.jl")
