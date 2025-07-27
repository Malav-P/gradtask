import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.1, momentum=0, modulo=None, noise=0, clip_grad_norm=None):

        self._parameters = parameters
        self._m = np.zeros_like(parameters)
        self._gamma = momentum
        self._lr = lr
        self._modulo = modulo
        self._noise = noise
        self._clip_grad_norm = np.abs(clip_grad_norm) if clip_grad_norm else clip_grad_norm

    def step(self, gradient):

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
    
    

if __name__ == "__main__":
    parameters = np.random.randn(2)
    gradient = np.random.randn(2)

    optimizer = SGD(parameters)
    optimizer.step(gradient)

    print(optimizer.parameters - parameters)

