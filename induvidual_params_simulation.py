import numpy as np
import matplotlib.pyplot as plt

def simulate_traffic(a, b, d, dt, T, x0, v0):
    steps = int(T / dt)
    t = np.linspace(0, T, steps + 1)

    a = np.array(a)
    b = np.array(b)
    d = np.array(d)

    x = np.zeros((3, steps + 1))
    v = np.zeros((3, steps + 1))

    x[:, 0] = x0
    v[:, 0] = v0

    for n in range(steps):
        x_prime = v[:, n]
        v_prime = np.zeros(3)

        # Car 1 follows Car 2
        v_prime[0] = a[0] * (x[1, n] - x[0, n] - d[0]) + b[0] * (v[1, n] - v[0, n])

        # Car 2 follows Car 3
        v_prime[1] = a[1] * (x[2, n] - x[1, n] - d[1]) + b[1] * (v[2, n] - v[1, n])

        # Car 3 follows a virtual point moving ahead at the average traffic speed
        virtual_x = x[2, n] + d[2]
        virtual_v = np.mean(v[:, n])

        v_prime[2] = a[2] * (virtual_x - x[2, n] - d[2]) + b[2] * (virtual_v - v[2, n])

        x[:, n + 1] = x[:, n] + dt * x_prime
        v[:, n + 1] = v[:, n] + dt * v_prime

    gap1 = x[1] - x[0]
    gap2 = x[2] - x[1]
    gap3 = np.full(steps + 1, d[2])

    return t, x, v, gap1, gap2, gap3


def plot_results(name, t, x, v, gap1, gap2, gap3, d):
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
    plt.axhline(d[0], linestyle="--", label="Desired gap Car 1")
    plt.axhline(d[1], linestyle=":", label="Desired gap Car 2")
    plt.axhline(d[2], linestyle="-.", label="Desired gap Car 3")
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
    plt.scatter([d[0]], [v[0, -1]], label="Car 1 desired state")
    plt.scatter([d[1]], [v[1, -1]], label="Car 2 desired state")
    plt.scatter([d[2]], [v[2, -1]], label="Car 3 desired state")
    plt.xlabel("Gap to object ahead")
    plt.ylabel("Velocity")
    plt.title(f"Phase Plot: {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_phase.png", dpi=200, bbox_inches="tight")
    plt.close()


def summarize_results(name, d, v, gap1, gap2, gap3):
    print(f"\nScenario: {name}")
    print(f"  Final velocities: car1={v[0,-1]:.3f}, car2={v[1,-1]:.3f}, car3={v[2,-1]:.3f}")
    print(f"  Min gaps: gap1={np.min(gap1):.3f}, gap2={np.min(gap2):.3f}, gap3={np.min(gap3):.3f}")
    print("  Mean abs gap error:")
    print(f"    car1={np.mean(np.abs(gap1 - d[0])):.3f}")
    print(f"    car2={np.mean(np.abs(gap2 - d[1])):.3f}")
    print(f"    car3={np.mean(np.abs(gap3 - d[2])):.3f}")


def run_simulation(scenario):
    name = scenario["name"]
    a = scenario["a"]
    b = scenario["b"]
    d = scenario["d"]
    dt = scenario.get("dt", 0.01)
    T = scenario.get("T", 30)
    x0 = scenario.get("x0", (0.0, 3.0, 6.0))
    v0 = scenario.get("v0", (0.8, 1.2, 1.0))

    results = simulate_traffic(a, b, d, dt, T, x0, v0)
    t, x, v, gap1, gap2, gap3 = results

    plot_results(name, t, x, v, gap1, gap2, gap3, d)
    summarize_results(name, d, v, gap1, gap2, gap3)

    print(f"Saved plots for {name}")


SCENARIOS = [
    {
        "name": "three_balanced_drivers",
        "a": [0.45, 0.45, 0.45],
        "b": [1.20, 1.20, 1.20],
        "d": [2.0, 2.0, 2.0],
    },
    {
        "name": "one_aggressive_driver",
        "a": [1.20, 0.45, 0.45],
        "b": [0.45, 1.20, 1.20],
        "d": [1.4, 2.3, 2.3],
    },
    {
        "name": "one_cautious_driver",
        "a": [0.35, 0.35, 0.35],
        "b": [1.40, 1.40, 1.40],
        "d": [3.2, 2.2, 2.2],
    },
    {
        "name": "mixed_driver_styles",
        "a": [1.10, 0.40, 0.70],
        "b": [0.50, 1.50, 0.90],
        "d": [1.5, 3.0, 2.2],
    },
    {
        "name": "tailgater_in_back",
        "a": [1.50, 0.55, 0.55],
        "b": [0.35, 1.20, 1.20],
        "d": [1.1, 2.4, 2.4],
    },
    {
        "name": "nervous_middle_driver",
        "a": [0.50, 1.60, 0.50],
        "b": [1.10, 0.35, 1.10],
        "d": [2.2, 3.2, 2.2],
    },
    {
        "name": "calm_absorbing_middle_driver",
        "a": [0.90, 0.25, 0.90],
        "b": [0.60, 1.80, 0.60],
        "d": [1.7, 3.4, 1.7],
    },
    {
        "name": "accordion",
        "a": [1.40, 1.10, 0.35],
        "b": [0.35, 0.45, 1.50],
        "d": [1.3, 1.6, 3.0],
    },
    {
        "name": "overcorrecting",
        "a": [1.70, 1.50, 1.30],
        "b": [0.30, 0.35, 0.40],
        "d": [1.5, 1.7, 1.9],
    },
    {
        "name": "highly_damped",
        "a": [0.30, 0.35, 0.40],
        "b": [1.80, 1.60, 1.50],
        "d": [3.5, 3.0, 2.7],
    },
    {
        "name": "leader_more_cautious_than_followers",
        "a": [1.10, 0.90, 0.25],
        "b": [0.55, 0.70, 1.80],
        "d": [1.4, 1.8, 3.6],
    },
    {
        "name": "leader_more_reactive_than_followers",
        "a": [0.35, 0.45, 1.40],
        "b": [1.40, 1.20, 0.45],
        "d": [2.8, 2.5, 1.6],
    },
]


for scenario in SCENARIOS:
    run_simulation(scenario)

print("All simulations complete.")