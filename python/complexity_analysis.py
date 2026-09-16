import matplotlib.pyplot as plt
import numpy as np

n_sims = np.linspace(1, 100, 100)
linear = n_sims
quadratic = n_sims**2
cubic = n_sims**3
exponential = 1.1**n_sims

plt.figure(figsize=(10, 7))
plt.plot(
    n_sims,
    linear,
    label="Monte Carlo Approach (Linear $O(N)$)",
    linewidth=3,
    color="green",
)
plt.plot(
    n_sims,
    quadratic,
    label="Quadratic Complexity $O(N^2)$",
    linestyle="--",
    color="orange",
)
plt.plot(
    n_sims,
    cubic,
    label="Cubic Complexity $O(N^3)$",
    linestyle="--",
    color="red",
)
plt.plot(
    n_sims,
    exponential,
    label="Exponential Complexity $O(a^N)$",
    linestyle=":",
    color="purple",
)
plt.ylim(0, 1000)
plt.title("Asymptotic Computational Cost")
plt.xlabel("Input Size ($N$)")
plt.ylabel("Computational Cost (Relative Time)")
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig("asymptotic_complexity.png")
plt.close()
