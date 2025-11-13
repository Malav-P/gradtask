import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.1, momentum=0, modulo=None, noise=0, clip_grad_norm=None,
                 cosine_anneal=False, t_max=None, eta_min=0.0, anneal_after=0):
        self._parameters = parameters
        self._m = np.zeros_like(parameters)
        self._gamma = momentum

        # learning rate / cosine annealing settings
        self._lr = lr
        self._initial_lr = lr  # keep original lr for annealing formula
        self._cosine_anneal = bool(cosine_anneal)
        self._t_max = t_max
        self._eta_min = eta_min
        self._step = 0  # global step counter
        self._anneal_after = int(anneal_after) if anneal_after is not None else 0
        if self._anneal_after < 0:
            raise ValueError("anneal_after must be non-negative")

        # legacy/internal anneal counter (used when annealing is active)
        self._t = 0

        if self._cosine_anneal and (self._t_max is None or self._t_max <= 0):
            raise ValueError("t_max must be a positive integer when cosine_anneal=True")

        self._modulo = modulo
        self._noise = noise
        self._clip_grad_norm = np.abs(clip_grad_norm) if clip_grad_norm else clip_grad_norm

    def step(self, gradient):
        # update global step counter
        self._step += 1

        # start cosine annealing only after anneal_after steps
        if self._cosine_anneal:
            if self._step <= self._anneal_after:
                # keep constant lr until anneal_after
                self._lr = self._initial_lr
            else:
                # annealing time index (starts at 1 when first anneal step happens)
                self._t = self._step - self._anneal_after
                T = self._t_max
                t = self._t
                lr0 = self._initial_lr
                # if we've passed the schedule length, set to eta_min
                if t >= T:
                    self._lr = self._eta_min
                else:
                    self._lr = self._eta_min + 0.5 * (lr0 - self._eta_min) * (1 + np.cos(np.pi * t / T))

        if self._clip_grad_norm:
            gradient = np.clip(gradient, -self._clip_grad_norm, self._clip_grad_norm)

        if self._noise > 0:
            gradient = self.add_noise(gradient)

        self._m = self._gamma * self._m + (1-self._gamma) * gradient
        self._parameters = self._parameters - self._lr * self._m

        if self._modulo:
            self._parameters = self._parameters % self._modulo

    def add_noise(self, gradient):
        """
        Make the gradients noisy (to avoid local minima). noisy_g = g * (1 + noise)

        Args:
            gradient (np.ndarray): array of gradients

        Returns:
            noised_gradient: array of noisy gradients of same shape as arr
        """

        noise = np.random.normal(loc=0, scale=self._noise, size=gradient.shape)

        noised_gradient = gradient * (1 + noise)

        return noised_gradient


    @property
    def parameters(self):
        return self._parameters

    @property
    def lr(self):
        "Return current learning rate."
        return self._lr
    

if __name__ == "__main__":
    parameters = np.random.randn(2)
    gradient = np.random.randn(2)

    optimizer = SGD(parameters, cosine_anneal=True, t_max=100, lr=0.1, eta_min=1e-4)
    optimizer.step(gradient)

    print(optimizer.parameters - parameters)

