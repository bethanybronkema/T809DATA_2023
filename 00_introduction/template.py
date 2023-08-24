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

def plot_three_normals():
    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1, -5, 5)
    plot_normal(1, 1.5, -5, 5)
    plt.show()
    plt.savefig(1_2_1)
plot_three_normals()

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    tot_sum = np.zeros(len(x))
    for i in range(len(sigmas)):
        in_par = (-np.power(x-mus[i], 2))/(2*np.power(sigmas[i], 2))
        coef = weights[i]/(np.sqrt(2*np.pi*np.power(sigmas[i], 2)))
        p1 = coef*np.exp(in_par)
        tot_sum = tot_sum + p1
    return tot_sum
print(normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3]))

'''
def _compare_components_and_mixture():
    # Part 2.2

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1

def _plot_mixture_and_samples():
    # Part 3.2

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
'''