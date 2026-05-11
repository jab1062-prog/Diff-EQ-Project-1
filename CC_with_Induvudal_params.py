import numpy as np
import matplotlib.pyplot as plt

def simulate_traffic_capacity_ratio(a, b, K, dt, T, x0, v0, min_gap=0.15):
    steps = int(T / dt)
    t = np.linspace(0, T, steps + 1)

    a = np.array(a)
    b = np.array(b)
    K = np.array(K)

    x = np.zeros((3, steps + 1))
    v = np.zeros((3, steps + 1))

    x[:, 0] = x0
    v[:, 0] = v0

    for n in range(steps):
        x_prime = v[:, n]
        v_prime = np.zeros(3)

        gap1 = max(x[1, n] - x[0, n], min_gap)
        gap2 = max(x[2, n] - x[1, n], min_gap)

        virtual_x = x[2, n] + K[2]
        virtual_v = np.mean(v[:, n])
        gap3 = max(virtual_x - x[2, n], min_gap)

        gaps = np.array([gap1, gap2, gap3])
        v_ahead = np.array([v[1, n], v[2, n], virtual_v])

        for i in range(3):
            spacing_force = gaps[i] / K[i] - 1
            damping_force = v_ahead[i] - v[i, n]
            v_prime[i] = a[i] * spacing_force + b[i] * damping_force

        x[:, n + 1] = x[:, n] + dt * x_prime
        v[:, n + 1] = v[:, n] + dt * v_prime

        # Prevent cars from reversing
        v[:, n + 1] = np.maximum(v[:, n + 1], 0)

        # Prevent cars from overlapping
        if x[1, n + 1] - x[0, n + 1] < min_gap:
            x[0, n + 1] = x[1, n + 1] - min_gap

        if x[2, n + 1] - x[1, n + 1] < min_gap:
            x[1, n + 1] = x[2, n + 1] - min_gap

    gap1 = x[1] - x[0]
    gap2 = x[2] - x[1]
    gap3 = np.full(steps + 1, K[2])

    return t, x, v, gap1, gap2, gap3


def plot_results(name, t, x, v, gap1, gap2, gap3, K):
    plt.figure(figsize=(8, 5))
    plt.plot(t, x[0], label="Car 1")
    plt.plot(t, x[1], label="Car 2")
    plt.plot(t, x[2], label="Car 3")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title(f"Position vs Time: {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_position.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(t, v[0], label="Car 1")
    plt.plot(t, v[1], label="Car 2")
    plt.plot(t, v[2], label="Car 3")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.title(f"Velocity vs Time: {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_velocity.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(t, gap1, label="Gap: Car 2 - Car 1")
    plt.plot(t, gap2, label="Gap: Car 3 - Car 2")
    plt.plot(t, gap3, label="Virtual gap for Car 3")
    plt.axhline(K[0], linestyle="--", label="K for Car 1")
    plt.axhline(K[1], linestyle=":", label="K for Car 2")
    plt.axhline(K[2], linestyle="-.", label="K for Car 3")
    plt.xlabel("Time")
    plt.ylabel("Gap")
    plt.title(f"Gap vs Time: {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_gap.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(gap1, v[0], label="Car 1")
    plt.plot(gap2, v[1], label="Car 2")
    plt.plot(gap3, v[2], label="Car 3")
    plt.scatter([K[0]], [v[0, -1]], label="Car 1 equilibrium")
    plt.scatter([K[1]], [v[1, -1]], label="Car 2 equilibrium")
    plt.scatter([K[2]], [v[2, -1]], label="Car 3 equilibrium")
    plt.xlabel("Gap to object ahead")
    plt.ylabel("Velocity")
    plt.title(f"Phase Plot: {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_phase.png", dpi=200, bbox_inches="tight")
    plt.close()


def summarize_results(name, K, v, gap1, gap2, gap3):
    print(f"\nScenario: {name}")
    print(f"  Final velocities: car1={v[0,-1]:.3f}, car2={v[1,-1]:.3f}, car3={v[2,-1]:.3f}")
    print(f"  Min gaps: gap1={np.min(gap1):.3f}, gap2={np.min(gap2):.3f}, gap3={np.min(gap3):.3f}")
    print("  Mean abs normalized gap error:")
    print(f"    car1={np.mean(np.abs(gap1 / K[0] - 1)):.3f}")
    print(f"    car2={np.mean(np.abs(gap2 / K[1] - 1)):.3f}")
    print(f"    car3={np.mean(np.abs(gap3 / K[2] - 1)):.3f}")


def run_simulation(scenario):
    name = scenario["name"]
    a = scenario["a"]
    b = scenario["b"]
    K = scenario["K"]

    dt = scenario.get("dt", 0.005)
    T = scenario.get("T", 25)
    x0 = scenario.get("x0", (0.0, 3.0, 6.0))
    v0 = scenario.get("v0", (0.8, 1.0, 1.0))
    min_gap = scenario.get("min_gap", 0.15)

    results = simulate_traffic_capacity_ratio(a, b, K, dt, T, x0, v0, min_gap)
    t, x, v, gap1, gap2, gap3 = results

    plot_results(name, t, x, v, gap1, gap2, gap3, K)
    summarize_results(name, K, v, gap1, gap2, gap3)

    print(f"Saved plots for {name}")

SCENARIOS = [
    {
        "name": "baseline_even_spacing",
        "a": [0.8, 0.8, 0.8],
        "b": [1.4, 1.4, 1.4],
        "K": [3.0, 3.0, 3.0],
        "x0": (0.0, 3.0, 6.0),
        "v0": (1.0, 1.0, 1.0),
        "T": 30,
        "dt": 0.01,
    },
    {
        "name": "gentle_compression_wave",
        "a": [1.0, 0.9, 0.8],
        "b": [1.1, 1.2, 1.3],
        "K": [2.4, 3.0, 3.6],
        "x0": (0.0, 3.0, 6.0),
        "v0": (0.9, 1.1, 1.0),
        "T": 35,
        "dt": 0.01,
    },
    {
        "name": "cautious_middle_driver",
        "a": [0.9, 0.6, 0.9],
        "b": [1.2, 1.7, 1.2],
        "K": [2.6, 4.2, 2.6],
        "x0": (0.0, 3.2, 6.4),
        "v0": (1.0, 0.9, 1.05),
        "T": 35,
        "dt": 0.01,
    },
    {
        "name": "tail_driver_closes_gap_smoothly",
        "a": [1.2, 0.8, 0.8],
        "b": [1.0, 1.4, 1.4],
        "K": [2.0, 3.2, 3.2],
        "x0": (0.0, 3.5, 7.0),
        "v0": (0.8, 1.0, 1.0),
        "T": 35,
        "dt": 0.01,
    },
    {
        "name": "open_road_cautious_group",
        "a": [0.6, 0.6, 0.6],
        "b": [1.8, 1.7, 1.6],
        "K": [4.0, 4.2, 4.4],
        "x0": (0.0, 3.0, 6.0),
        "v0": (1.0, 1.0, 1.0),
        "T": 40,
        "dt": 0.01,
    },
    {
    "name": "mixed_but_visible_drivers",
    "a": [1.7, 0.8, 1.2],
    "b": [0.7, 1.5, 0.9],
    "K": [1.8, 4.8, 2.6],
    "x0": (0.0, 2.0, 7.5),
    "v0": (0.4, 1.6, 0.9),
    "T": 35,
    "dt": 0.008,
    }
]

for scenario in SCENARIOS:
    run_simulation(scenario)

print("All capacity-ratio simulations complete.")