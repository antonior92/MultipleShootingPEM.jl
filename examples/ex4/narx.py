import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from base import GenerateData


def one_step_ahead(u, y, x0, N, ny, nu, params):
    state = x0
    y_pred = []
    for i in range(N):
        # Compute next value
        y_next = jnp.dot(params, state)
        y_pred.append(y_next)
        # assemble next state vector
        y1 = y[i + ny]
        y2 = y[i + ny - 1]
        u1 = u[i + nu]
        state = jnp.stack([y1, y2, u1])
    return jnp.stack(y_pred)


def solve_osa(u, y, x0, N, ny, nu, params0, verbose=0):
    params0 = np.asarray(params0)
    fjac_1stepa = jax.jit(jax.jacfwd(partial(one_step_ahead, u, y, x0, N, ny, nu)))
    jac_1stepa = np.asarray(fjac_1stepa(params0))

    est_param, _, _, _ = np.linalg.lstsq(jac_1stepa, y[-N:], rcond=None)
    return est_param, 1


if __name__ =='__main__':
    import matplotlib.pyplot as plt
    gn = GenerateData()
    u, y, x0 = gn.generate(0)
    N, ny, nu, = gn.N, gn.ny, gn.nu
    y_pred = one_step_ahead(u, y, x0, N, ny, nu, gn.theta)
    plt.plot(y_pred)
    plt.plot(y)
    plt.show()

    print(solve_osa(u, y, x0, N, ny, nu, [0., 0., 0.]))



