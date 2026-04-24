import numpy as np
import matplotlib.pyplot as plt

def run_simulation(a, b, d, name):
    dt = 0.01
    T = 30
    steps = int(T / dt)
    t = np.linspace(0, T, steps + 1)

    x1 = np.zeros(steps + 1)
    v1 = np.zeros(steps + 1)
    x2 = np.zeros(steps + 1)
    v2 = np.zeros(steps + 1)
    x3 = np.zeros(steps + 1)
    v3 = np.zeros(steps + 1)

    # Clean initial conditions
    x1[0] = 0.0
    x2[0] = 4.0
    x3[0] = 8.0

    v1[0] = 0.0
    v2[0] = 0.0
    v3[0] = 1.0

    for n in range(steps):
        x1_prime = v1[n]
        v1_prime = a * (x2[n] - x1[n] - d) + b * (v2[n] - v1[n])

        x2_prime = v2[n]
        v2_prime = a * (x3[n] - x2[n] - d) + b * (v3[n] - v2[n])

        x3_prime = v3[n]
        v3_prime = 0.0

        x1[n+1] = x1[n] + dt * x1_prime
        v1[n+1] = v1[n] + dt * v1_prime

        x2[n+1] = x2[n] + dt * x2_prime
        v2[n+1] = v2[n] + dt * v2_prime

        x3[n+1] = x3[n] + dt * x3_prime
        v3[n+1] = v3[n] + dt * v3_prime

    gap1 = x2 - x1
    gap2 = x3 - x2

    # Position plot
    plt.figure(figsize=(8, 5))
    plt.plot(t, x1, label="Car 1")
    plt.plot(t, x2, label="Car 2")
    plt.plot(t, x3, label="Car 3 Lead")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title(f"Position vs Time: {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_position.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Velocity plot
    plt.figure(figsize=(8, 5))
    plt.plot(t, v1, label="Car 1")
    plt.plot(t, v2, label="Car 2")
    plt.plot(t, v3, label="Car 3 Lead")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.title(f"Velocity vs Time: {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_velocity.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Gap plot
    plt.figure(figsize=(8, 5))
    plt.plot(t, gap1, label="Gap: Car 2 - Car 1")
    plt.plot(t, gap2, label="Gap: Car 3 - Car 2")
    plt.axhline(d, linestyle="--", label="Desired gap d")
    plt.xlabel("Time")
    plt.ylabel("Gap")
    plt.title(f"Gap vs Time: {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_gap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Phase plot
    plt.figure(figsize=(6, 6))
    plt.plot(gap1, v1, label="Car 1 phase path")
    plt.scatter([d], [v3[0]], label="Equilibrium")
    plt.xlabel("Gap to car ahead")
    plt.ylabel("Velocity")
    plt.title(f"Phase Plot: {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_phase.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plots for {name}")


# ---------------------------
# EXPERIMENTS
# ---------------------------

run_simulation(a=0.3, b=1.2, d=2.0, name="smooth_traffic")

run_simulation(a=1.2, b=0.4, d=2.0, name="oscillating_traffic")

run_simulation(a=0.5, b=1.0, d=4.0, name="larger_spacing")

print("All simulations complete.")