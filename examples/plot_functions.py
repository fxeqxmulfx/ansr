"""Generate 3D surface plots for benchmark functions."""
import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def sphere(x, y):
    return (x**2 + y**2) / 50.0


def shubert(x, y):
    i = np.arange(1, 6).reshape(1, 1, -1)
    x3 = x[:, :, np.newaxis]
    y3 = y[:, :, np.newaxis]
    sx = np.sum(i * np.cos((i + 1) * x3 + i), axis=2)
    sy = np.sum(i * np.cos((i + 1) * y3 + i), axis=2)
    result = sx * sy
    return (result - (-186.7309)) / (210.0 - (-186.7309))


def hilly(x, y):
    result = (
        20.0 + x**2 + y**2
        - 10.0 * np.cos(2.0 * math.pi * x)
        - 10.0 * np.cos(2.0 * math.pi * y)
        - 30.0 * np.exp(-((x - 1.0) ** 2 + y**2) / 0.1)
        + 200.0
        * np.exp(
            -((x + math.pi * 0.47) ** 2 + (y - math.pi * 0.2) ** 2) / 0.1
        )
        + 100.0 * np.exp(-((x - 0.5) ** 2 + (y + 0.5) ** 2) / 0.01)
        - 60.0 * np.exp(-((x - 1.33) ** 2 + (y - 2.0) ** 2) / 0.02)
        - 40.0 * np.exp(-((x + 1.3) ** 2 + (y + 0.2) ** 2) / 0.5)
        + 60.0 * np.exp(-((x - 1.5) ** 2 + (y + 1.5) ** 2) / 0.1)
    )
    result = -result
    return (result - (-229.91931214214105)) / (
        39.701816104859866 - (-229.91931214214105)
    )


functions = [
    ("sphere", sphere, (-5, 5), (-5, 5)),
    ("shubert", shubert, (-10, 10), (-10, 10)),
    ("hilly", hilly, (-3, 3), (-3, 3)),
]

for name, func, (xmin, xmax), (ymin, ymax) in functions:
    x = np.linspace(xmin, xmax, 200)
    y = np.linspace(ymin, ymax, 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9, rcount=100, ccount=100)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title(f"{name} (single pair, scaled to [0, 1])")
    fig.savefig(f"images/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved images/{name}.png")
