import numpy as np

THETA1 = [1.5, -0.7, 0.5]  # tau = 0.75
THETA2 = [0.5, -0.2, 2]  # tau = 0.25
THETA3 = [1.8, -0.95, 0.1]  # tau = 0.9

class GenerateData():
    def __init__(self,  N=300, ny=2, nu=1,noise_std=0.0, time_constant='interm'):
        self.N = N
        self.ny = ny
        self.nu = nu
        self.noise_std = noise_std
        self.n_max = max(nu, ny)
        if time_constant == 'interm':
            self.theta = np.asarray(THETA1)
            self.hold = 10
        elif time_constant == 'fast':
            self.theta = np.asarray(THETA2)
            self.hold = 3
        elif time_constant == 'slow':
            self.theta = np.asarray(THETA3)
            self.hold = 20

    def generate(self, seed):
        rng = np.random.RandomState(seed)
        y = np.zeros(self.N+self.ny)  # Output with zeros placeholder
        u = np.repeat(rng.normal(size=int(np.ceil((self.N+self.nu)/self.hold))), self.hold)[:self.N+self.nu]  # Random gausian noise with hold
        for i in range(self.N):
            y1 = y[i+self.ny-1]
            y2 = y[i+self.ny-2]
            u1 = u[i+self.nu-1]
            x = [y1, y2, u1]
            y_next = np.dot(self.theta, x)
            y[i+self.ny] = y_next
        # Define initial states
        x0 = np.asarray([y[1], y[0], u[0]])
        # Add output white noise
        v = self.noise_std * rng.normal(size=self.N+self.ny)
        y += v
        return u, y, x0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    gn = GenerateData(time_constant='fast')
    u, y, x0 = gn.generate(0)
    plt.plot(u)
    plt.plot(y)
    plt.show()