module MultipleShootingPEM

using LossFunctions
using PyCall

const scipy_opt = PyNULL()
const scipy_sps = PyNULL()
const scipy_spslin = PyNULL()

function __init__()
    copy!(scipy_opt, pyimport_conda("scipy.optimize", "scipy"))
    copy!(scipy_sps, pyimport_conda("scipy.sparse", "scipy"))
    copy!(scipy_spslin, pyimport_conda("scipy.sparse.linalg", "scipy"))
end

include("util.jl")
include("sensitivity_equations.jl")
include("space_state_model.jl")
include("one_shoot_simulation.jl")
include("multiple_shooting.jl")
include("optimization_problem.jl")

end # module
