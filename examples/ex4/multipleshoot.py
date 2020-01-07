import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator

def shoot(u, N, nu, shoot_len, index, x0, params):
    y_pred = []
    y1 = x0[0]
    y2 = x0[1]
    u1 = u[index + nu - 1]
    state = jnp.stack([y1, y2, u1])
    for j in range(min(shoot_len, N-index)):
        i = j + index
        # Compute next value
        y_next = jnp.dot(params, state)
        y_pred.append(y_next)
        # propagate state
        y1 = y_next
        y2 = state[0]
        u1 = u[i + nu]
        state = jnp.stack([y1, y2, u1])
    state = jnp.stack([state[0], state[1]])
    y_pred = jnp.stack(y_pred)
    return y_pred, state


fjac_shoot = jax.jacfwd(shoot, argnums=(-2, -1))


class MultipleShooting():

    def __init__(self, u, y, N, ny, nu, shoot_len):
        self.u, self.y = u, y
        self.N, self.ny, self.nu = N, ny, nu
        self.shoot_len = shoot_len
        x0s = []
        for i in range(N):
            if i % shoot_len == 0:  # Reinitialize state
                x0s.append(y[i + 1])
                x0s.append(y[i])
        self.x0s = np.stack(x0s)
        self.n_x0s = len(x0s)
        self.n_ext_params = 3 + len(x0s)

    def ext_params(self, x0s, params):
        return jnp.concatenate([x0s, params])

    def split_params(self, ext_params):
        ny, nu = self.ny, self.nu
        params = ext_params[-(nu+ny):]
        x0s = ext_params[:-(nu+ny)]
        return x0s, params

    def predict(self, x0s, params):
        y_pred = []
        constr = []
        index_state = 0
        for i in range(self.N):
            if i % self.shoot_len == 0:   # Reinitialize state
                x0 = [x0s[index_state], x0s[index_state + 1]]
                y_pred_shoot, state_next = shoot(self.u, self.N, self.nu, self.shoot_len, i, x0, params)
                index_state += 2
                y_pred.append(y_pred_shoot)
                if index_state < self.n_x0s:
                    constr.append(jnp.stack([x0s[index_state], x0s[index_state + 1]]) - state_next)

        return y_pred, constr

    def jacobian(self, x0s, params):
        jac = []
        jac_states = []
        index_state = 0
        for i in range(self.N):
            if i % self.shoot_len == 0:   # Reinitialize state
                x0 = jnp.stack([x0s[index_state], x0s[index_state + 1]])
                jac_shoot, jac_constr_shoot = fjac_shoot(self.u, self.N, self.nu, self.shoot_len, i, x0, params)
                jac.append(jac_shoot)
                if index_state < self.n_x0s:
                    jac_states.append(jac_constr_shoot)

        return jac, jac_states

    def cost_func(self, ext_params):
        x0s, params = self.split_params(ext_params)
        y_pred, _ = self.predict(x0s, params)
        y_pred = jnp.concatenate(y_pred)
        return jnp.sum((y_pred - self.y[-N:]) * (y_pred - self.y[-N:]))

    def constr(self, ext_params):
        x0s, params = self.split_params(ext_params)
        _, constr = self.predict(x0s, params)
        constr = jnp.concatenate(y_pred)
        return constr

    def grad(self, ext_params):
        x0s, params = self.split_params(ext_params)
        y_pred, _ = self.predict(x0s, params)
        jac, _ = self.jacobian(x0s, params)
        jac = np.concatenate([jnp.concatenate(j, axis=1) for j in jac], axis=0)
        jac_x0s, jac_params = tuple(zip(*jac))
        jac_params = jnp.concatenate(jac_params, axis=0)
        grad_params = jac_params.T.dot(jnp.concatenate(y_pred))
        grad_x0 = jnp.concatenate([jac_x0s[i].T.dot(y_pred[i]) for i in range(len(y_pred))])
        grad = ext_params(grad_x0, grad_params)
        return grad

    def hess(self, ext_params):
        x0s, params = self.split_params(ext_params)
        jac, _ = self.jacobian(x0s, params)
        jac = np.concatenate([jnp.concatenate(j, axis=1) for j in jac], axis=0)
        jac_x0s, jac_params = tuple(zip(*jac))
        jac_params = jnp.concatenate(jac_params, axis=0)

        def hessp(ext_p):
            p_x0s, p_params = self.split_params(ext_p)
            hessp_x0s = jnp.concatenate(jac_x0s[i].T.dot(jac_x0s[i].dot(jnp.stack(p_x0s[i], p_x0s[i+1]))) for i in range(p_x0s, step=2)])
            hessp_params = jac_params.T.dot(jac_params.dot(p_params))
            return ext_params(hessp_x0s, hessp_params)

        return LinearOperator((self.n_ext_params, self.n_ext_params), matvec(hessp))


if __name__ =='__main__':
    import matplotlib.pyplot as plt
    from base import GenerateData
    gn = GenerateData()
    u, y, x0 = gn.generate(0)
    N, ny, nu, = gn.N, gn.ny, gn.nu

    shoot_len = 1

    # Instanciate
    ms = MultipleShooting(u, y, N, ny, nu, shoot_len)
    # Get x0s
    x0s = ms.x0s
    # Check
    print('n_x0s = {}, n_states = {}, datasize = {}'.format(len(x0s), ny, N))
    # Predict
    y_pred, c = ms.predict(x0s, gn.theta)
    # Check
    print('output size = {}, n_const = {}'.format(len(y_pred), len(c)))
    plt.plot(y_pred)
    plt.plot(y)
    plt.show()
    # Compute cost function
    cost = ms.cost_func(ms.ext_params(x0s, gn.theta))
    print('cost = {}'.format(cost))
    # Compute constr
    constr = ms.constr(ms.ext_params(x0s, gn.theta))
    print('constr len = {}'.format(len(constr)))
    # Check jacobian
    jac, jac_states = ms.jacobian(x0s, gn.theta)
    print('jac len = {}, jac states len = {}'.format(len(jac), len(jac_states)))
    # Check grad
    #grad = ms.grad(ms.ext_params(x0s, gn.theta))
    #print('grad len = {}'.format(len(grad)))