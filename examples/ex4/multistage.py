import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from base import GenerateData
from scipy.optimize import least_squares


def multi_step_ahead(u, y, x0, N, ny, nu, n_stages, params):
    state = x0
    y_pred = []
    for i in range(N):
        y_next = jnp.dot(params, state)
        y_pred.append(y_next)
        # assemble next state vector
        y1 = y[i - n_stages + ny] if i - n_stages + ny >= 0 else 0
        y2 = y[i - n_stages + ny - 1] if i - n_stages + ny - 1 >= 0 else 0
        u1 = u[i - n_stages + nu] if i - n_stages + nu >= 0 else 0
        state = jnp.stack([y1, y2, u1])
        for j in range(n_stages):
            y_next = jnp.dot(params, state)
            y1 = y_next
            y2 = state[0]
            u1 = u[i - n_stages + j + 1 + nu] if i - n_stages + j + 1 + nu >= 0 else 0
            state = jnp.stack([y1, y2, u1])
    return jnp.stack(y_pred)


def solve_msa(u, y, x0, N, ny, nu, n_stages, params0, verbose=0):
    params0 = np.asarray(params0)
    fun = lambda params: multi_step_ahead(u, y, x0, N, ny, nu, n_stages, params) - y[-N:]
    jac = jax.jacfwd(partial(multi_step_ahead, u, y, x0, N, ny, nu, n_stages))
    sol = least_squares(fun, params0, jac, verbose=verbose)
    return sol['x']


if __name__ == '__main__':
    gn = GenerateData()
    u, y, x0 = gn.generate(0)
    N, ny, nu, = gn.N, gn.ny, gn.nu
    n_stages = 10
    y_pred = multi_step_ahead(u, y, x0, N, ny, nu, n_stages, gn.theta, verbose=2)

    plt.plot(y_pred)
    plt.plot(y)
    plt.show()

    sol = solve_msa(u, y, x0, N, ny, nu, n_stages, [0.0, 0.0, 0.0])
    print(sol)
