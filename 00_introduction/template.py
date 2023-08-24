import numpy as np
import matplotlib.pyplot as plt

def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    d1 = 2*np.power(sigma,2)
    n1 = -np.power(x-mu,2)
    d2 = np.sqrt(2*np.pi*np.power(sigma,2))
    p = (1/d2)*np.exp(n1/d1)
    return p

def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    x = np.linspace(x_start, x_end, num=500)
    plt.plot(x,normal(x, sigma, mu))

def _plot_three_normals():
    # Part 1.2

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1

def _compare_components_and_mixture():
    # Part 2.2

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1

def _plot_mixture_and_samples():
    # Part 3.2

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
'''