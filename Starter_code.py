import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# PARAMETERS
# ---------------------------
a = 1      # spacing sensitivity
b = 1.0      # speed sensitivity
d = 2.0      # desired gap
dt = 0.01    # time step
T = 30       # total simulation time
steps = int(T / dt)

# ---------------------------
# ARRAYS TO STORE SOLUTION
# ---------------------------
t = np.linspace(0, T, steps + 1)

x1 = np.zeros(steps + 1)
v1 = np.zeros(steps + 1)

x2 = np.zeros(steps + 1)
v2 = np.zeros(steps + 1)

x3 = np.zeros(steps + 1)
v3 = np.zeros(steps + 1)

# ---------------------------
# INITIAL CONDITIONS
# ---------------------------
x1[0] = float(input("Enter initial position of car 1 [lead car] (e.g. 0.0): "))
x2[0] = float(input("Enter initial position of car 2 (e.g. 0.0): "))
x3[0] = float(input("Enter initial position of car 3 (e.g. 0.0): "))

v1[0] = float(input("Enter initial velocity of car 1 [lead car] (e.g. 1.0): "))
v2[0] = float(input("Enter initial velocity of car 2 (e.g. 1.0): "))
v3[0] = float(input("Enter initial velocity of car 3 (e.g. 1.0): ")) 

# ---------------------------
# EULER METHOD
# ---------------------------
for n in range(steps):
    # derivatives at current step
    x1_prime = v1[n]
    v1_prime = a * (x2[n] - x1[n] - d) + b * (v2[n] - v1[n])

    x2_prime = v2[n]
    v2_prime = a * (x3[n] - x2[n] - d) + b * (v3[n] - v2[n])

    x3_prime = v3[n]
    v3_prime = 0.0   # lead car keeps constant velocity

    # Euler updates
    x1[n+1] = x1[n] + dt * x1_prime
    v1[n+1] = v1[n] + dt * v1_prime

    x2[n+1] = x2[n] + dt * x2_prime
    v2[n+1] = v2[n] + dt * v2_prime

    x3[n+1] = x3[n] + dt * x3_prime
    v3[n+1] = v3[n] + dt * v3_prime

# ---------------------------
# GAPS
# ---------------------------
gap1 = x2 - x1
gap2 = x3 - x2

# ---------------------------
# PLOT 1: POSITION VS TIME
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(t, x1, label="Car 1")
plt.plot(t, x2, label="Car 2")
plt.plot(t, x3, label="Car 3 (lead)")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Position vs Time")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# PLOT 2: VELOCITY VS TIME
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(t, v1, label="Car 1")
plt.plot(t, v2, label="Car 2")
plt.plot(t, v3, label="Car 3 (lead)")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Velocity vs Time")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# PLOT 3: GAP VS TIME
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(t, gap1, label="Gap: Car 2 - Car 1")
plt.plot(t, gap2, label="Gap: Car 3 - Car 2")
plt.axhline(d, linestyle='--', label="Desired gap d")
plt.xlabel("Time")
plt.ylabel("Gap")
plt.title("Gap vs Time")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# PLOT 4: PHASE PLOT
# ---------------------------
plt.figure(figsize=(6,6))
plt.plot(gap1, v1, label="Car 1 phase path")
plt.xlabel("Gap to car ahead")
plt.ylabel("Velocity")
plt.title("Phase Plot: gap vs velocity")
plt.legend()
plt.grid(True)
plt.show()