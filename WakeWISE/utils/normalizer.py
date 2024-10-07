import numpy as np

class Normalizer:

    def __init__(self, a: float, b: float, type='min_max') -> None:

        assert type in ['min_max', 'mean_std'], f'Error: Normalization type {type} not supported'

        self.a = a # min or mean
        self.b = b # max or std
        self.type = type

    def normalize(self, x: np.ndarray) -> np.ndarray:

        if self.type == 'min_max':
            return (x - self.a) / (self.b - self.a)
        elif self.type == 'mean_std':
            return (x - self.a) / self.b
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:

        if self.type == 'min_max':
            return x * (self.b - self.a) + self.a
        elif self.type == 'mean_std':
            return x * self.b + self.a
    

if __name__ == "__main__":

    n = Normalizer(0, 10)

    print(n.denormalize(n.normalize(5)))
    print(n.denormalize(n.normalize(0)))
    print(n.denormalize(n.normalize(10)))
    print(n.denormalize(n.normalize(15)))