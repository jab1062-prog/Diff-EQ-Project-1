import numpy as np
import matplotlib.pyplot as plt

def simulate_traffic(a, b, d, dt, T, x0, v0, lead_velocity_fn):
    steps = int(T / dt)
    t = np.linspace(0, T, steps + 1)

    x1 = np.zeros(steps + 1)
    v1 = np.zeros(steps + 1)
    x2 = np.zeros(steps + 1)
    v2 = np.zeros(steps + 1)
    x3 = np.zeros(steps + 1)
    v3 = np.zeros(steps + 1)

    x1[0], x2[0], x3[0] = x0
    v1[0], v2[0], v3[0] = v0

    for n in range(steps):
        current_t = t[n]
        next_t = t[n + 1]

        # Prescribed lead-car profile lets us test richer scenarios.
        v3[n] = lead_velocity_fn(current_t)
        x1_prime = v1[n]
        v1_prime = a * (x2[n] - x1[n] - d) + b * (v2[n] - v1[n])

        x2_prime = v2[n]
        v2_prime = a * (x3[n] - x2[n] - d) + b * (v3[n] - v2[n])

        x3_prime = v3[n]

        x1[n+1] = x1[n] + dt * x1_prime
        v1[n+1] = v1[n] + dt * v1_prime

        x2[n+1] = x2[n] + dt * x2_prime
        v2[n+1] = v2[n] + dt * v2_prime

        v3[n+1] = lead_velocity_fn(next_t)
        x3[n+1] = x3[n] + dt * x3_prime

    gap1 = x2 - x1
    gap2 = x3 - x2
    return t, x1, v1, x2, v2, x3, v3, gap1, gap2

def plot_results(name, t, x1, v1, x2, v2, x3, v3, gap1, gap2, d):
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

def summarize_results(name, d, v1, v2, gap1, gap2):
    print(f"\nScenario: {name}")
    print(f"  Final velocities: car1={v1[-1]:.3f}, car2={v2[-1]:.3f}")
    print(f"  Min gaps: gap1={np.min(gap1):.3f}, gap2={np.min(gap2):.3f}")
    print(f"  Mean abs gap error: {np.mean(np.abs(gap1 - d)):.3f}")

def constant_lead(speed):
    return lambda _t: speed

def sinusoidal_lead(base_speed, amplitude, frequency_hz):
    return lambda t: base_speed + amplitude * np.sin(2.0 * np.pi * frequency_hz * t)

def braking_lead(base_speed, brake_start, brake_duration, slow_speed):
    def profile(t):
        if brake_start <= t <= brake_start + brake_duration:
            return slow_speed
        return base_speed
    return profile

def run_simulation(scenario):
    name = scenario["name"]
    a = scenario["a"]
    b = scenario["b"]
    d = scenario["d"]
    dt = scenario.get("dt", 0.01)
    T = scenario.get("T", 30)
    x0 = scenario.get("x0", (0.0, 4.0, 8.0))
    v0 = scenario.get("v0", (0.0, 0.0, 1.0))
    lead_velocity_fn = scenario.get("lead_velocity_fn", constant_lead(v0[2]))

    results = simulate_traffic(
        a=a,
        b=b,
        d=d,
        dt=dt,
        T=T,
        x0=x0,
        v0=v0,
        lead_velocity_fn=lead_velocity_fn,
    )
    plot_results(name, *results, d)
    summarize_results(name, d, results[2], results[4], results[7], results[8])
    print(f"Saved plots for {name}")


# ---------------------------
# EXPERIMENTS
# ---------------------------

SCENARIOS = [
    {
        "name": "smooth_traffic",
        "a": 0.3,
        "b": 1.2,
        "d": 2.0,
        "lead_velocity_fn": constant_lead(1.0),
    },
    {
        "name": "aggressive_following",
        "a": 1.2,
        "b": 0.4,
        "d": 2.0,
        "lead_velocity_fn": constant_lead(1.0),
    },
    {
        "name": "larger_spacing",
        "a": 0.5,
        "b": 1.0,
        "d": 4.0,
        "lead_velocity_fn": constant_lead(1.0),
    },
    {
        "name": "wave_lead_vehicle",
        "a": 0.6,
        "b": 1.1,
        "d": 2.5,
        "lead_velocity_fn": sinusoidal_lead(base_speed=1.0, amplitude=0.4, frequency_hz=0.08),
    },
    {
        "name": "lead_braking_event",
        "a": 0.8,
        "b": 1.3,
        "d": 2.5,
        "lead_velocity_fn": braking_lead(base_speed=1.2, brake_start=10.0, brake_duration=5.0, slow_speed=0.4),
    },
]

for scenario in SCENARIOS:
    run_simulation(scenario)

print("All simulations complete.")