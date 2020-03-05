import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from scipy.optimize import minimize, NonlinearConstraint
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import coo_matrix, eye, bmat
from scipy.optimize._numdiff import approx_derivative


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


fshoot = jax.jit(shoot, static_argnums=(0, 1, 2, 3, 4))


fjac_shoot = jax.jit(jax.jacfwd(shoot, argnums=(-2, -1)), static_argnums=(0, 1, 2, 3, 4))


class MultipleShooting():

    def __init__(self, u, y, N, ny, nu, shoot_len):
        self.u, self.y = jnp.array(u), jnp.array(y)
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
        self.saved_ext_params = None
        self.error, self.c = None, None
        self.jac_out, self.jac_states = None, None

    def ext_params(self, x0s, params):
        return jnp.concatenate([x0s, params])

    def split_params(self, ext_params):
        ny, nu = self.ny, self.nu
        params = jnp.array(ext_params[-(nu+ny):])
        x0s = jnp.array(ext_params[:-(nu+ny)])
        return x0s, params

    def predict(self, x0s, params):
        error = []
        constr = []
        index_state = 0
        for i in range(self.N):
            if i % self.shoot_len == 0:   # Reinitialize state
                x0 = jnp.stack([x0s[index_state], x0s[index_state + 1]])
                y_pred_shoot, state_next = fshoot(self.u, self.N, self.nu, self.shoot_len, i, x0, params)
                index_state += 2
                error.append(y_pred_shoot - self.y[i+self.ny:i+self.ny+self.shoot_len])
                if index_state < self.n_x0s:
                    constr.append(state_next - jnp.stack([x0s[index_state], x0s[index_state + 1]]))

        return error, constr

    def jacobian(self, x0s, params):
        jac = []
        jac_states = []
        index_state = 0
        for i in range(self.N):
            if i % self.shoot_len == 0:   # Reinitialize state
                x0 = jnp.stack([x0s[index_state], x0s[index_state + 1]])
                jac_shoot, jac_states_shoot = fjac_shoot(self.u, self.N, self.nu, self.shoot_len, i, x0, params)
                jac.append(jac_shoot)
                index_state += 2
                if index_state < self.n_x0s:
                    jac_states.append(jac_states_shoot)

        return jac, jac_states

    def compute(self, ext_params):
        if not np.equal(ext_params, self.saved_ext_params).all():
            x0s, params = self.split_params(ext_params)
            self.error, self.c = self.predict(x0s, params)
            self.jac_out, self.jac_states = self.jacobian(x0s, params)
            self.saved_ext_params = ext_params

    def cost_func(self, ext_params):
        self.compute(ext_params)
        error = jnp.concatenate(self.error)
        mse = error.dot(error)
        return mse

    def grad(self, ext_params):
        self.compute(ext_params)
        jac_x0s, jac_params = tuple(zip(*self.jac_out))
        jac_params = jnp.concatenate(jac_params, axis=0)
        grad_params = jac_params.T.dot(jnp.concatenate(self.error))
        grad_x0 = jnp.concatenate([jac_x0s[i].T.dot(self.error[i]) for i in range(len(self.error))])
        grad = self.ext_params(grad_x0, grad_params)
        return grad

    def hess(self, ext_params):
        self.compute(ext_params)
        jac_x0s, jac_params = tuple(zip(*self.jac_out))
        jac_params = jnp.concatenate(jac_params, axis=0)

        def matvec(ext_p):
            p_x0s, p_params = self.split_params(ext_p)
            hessp_x0s = jnp.concatenate(
                [jac_x0s[i].T.dot(jac_x0s[i].dot(jnp.stack([p_x0s[2*i], p_x0s[2*i+1]]))) for i in range(len(jac_x0s))])
            hessp_params = jac_params.T.dot(jac_params.dot(p_params))
            return self.ext_params(hessp_x0s, hessp_params)

        return LinearOperator((self.n_ext_params, self.n_ext_params), matvec=matvec)

    def constr(self, ext_params):
        self.compute(ext_params)
        constr = jnp.concatenate(self.c)
        return constr

    def jac(self, ext_params):
        self.compute(ext_params)
        # Assemble Jac
        jac = [
            [None] * i +
            [coo_matrix(self.jac_states[i][0])] +
            [-eye(2)] +
            [None] * (self.n_x0s//2 - 1 - i) +
            [coo_matrix(self.jac_states[i][1])]
            for i in range(self.n_x0s//2 - 1)]
        return bmat(jac)


def solve_ms(u, y, N, ny, nu, shoot_len, params0, verbose=0, initial_constr_penalty=1, initial_trust_radius=1):
    ms = MultipleShooting(u, y, N, ny, nu, shoot_len)

    initial_ext_params = ms.ext_params(ms.x0s,  jnp.stack(params0))
    nl_constr = NonlinearConstraint(ms.constr, 0, 0, ms.jac, '2-point')
    result = minimize(ms.cost_func, initial_ext_params, jac=ms.grad, hess=ms.hess,
                      constraints=nl_constr, method='trust-constr',
                      options={'verbose': verbose, 'initial_constr_penalty': initial_constr_penalty,
                               'initial_tr_radius': initial_trust_radius})

    return result['x'][-3:]


if __name__ =='__main__':
    from base import GenerateData
    gn = GenerateData(noise_std=0.0, time_constant='slow')
    u, y, x0 = gn.generate(28)
    N, ny, nu, = gn.N, gn.ny, gn.nu
    shoot_len = 2
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
    # Compute cost function
    cost = ms.cost_func(ms.ext_params(x0s, gn.theta))
    # Compute constr
    constr = ms.constr(ms.ext_params(x0s, gn.theta))
    print('constr len = {}'.format(len(constr)))
    # Check jacobian
    jac, jac_states = ms.jacobian(x0s, gn.theta)
    print('jac len = {}, jac states len = {}'.format(len(jac), len(jac_states)))
    # Check grad
    grad = ms.grad(ms.ext_params(x0s, gn.theta))
    print('grad len = {}'.format(len(grad)))
    # Check hess
    hessp = ms.hess(ms.ext_params(x0s, gn.theta))
    print('hessp len = {}'.format(len(hessp.dot(ms.ext_params(x0s, gn.theta)))))
    # Check Jacobian
    jac = ms.jac(ms.ext_params(x0s + 0.001, gn.theta + 0.01))
    jac_array = jac.toarray()
    #approx_jac = approx_derivative(ms.constr, ms.ext_params(x0s + 0.001, gn.theta+ 0.01), method='3-point')
    #print('jac shape', jac.shape)
    # Test solve
    params0 = jnp.stack([0., 0., 0.])
    x = solve_ms(u, y, N, ny, nu, shoot_len, params0, verbose=2)
    print(x)
