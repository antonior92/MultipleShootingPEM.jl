import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from base import GenerateData
from scipy.optimize import least_squares


def free_run_simulation(u, _, x0, N, ny, nu,  params):
    state = x0
    y_pred = []
    for i in range(N):
        # Compute next value
        y_next = jnp.dot(params, state)
        y_pred.append(y_next)
        # assemble next state vector
        y1 = y_next
        y2 = state[0]
        u1 = u[i+nu]
        state = jnp.stack([y1, y2, u1])
    return jnp.stack(y_pred)


def solve_frs(u, y, x0, N, ny, nu, params0, verbose=0):
    params0 = np.asarray(params0)
    fun = lambda params: free_run_simulation(u, y, x0, N, ny, nu, params) - y[-N:]
    jac = jax.jacfwd(partial(free_run_simulation, u, y, x0, N, ny, nu))
    sol = least_squares(fun, params0, jac, verbose=verbose)
    return sol['x']


if __name__ == '__main__':
    gn = GenerateData()
    u, y, x0 = gn.generate(0)
    N, ny, nu, = gn.N, gn.ny, gn.nu
    y_pred = free_run_simulation(u, y, x0, N, ny, nu, gn.theta, verbose=2)

    plt.plot(y_pred)
    plt.plot(y)
    plt.show()

    sol = solve_frs(u, y, x0, N, ny, nu, [0.0]*3)
    print(sol)
