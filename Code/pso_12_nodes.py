# for Medium sized  node PSO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# =====================================================
# 1. LOAD DATA
# =====================================================
nodes_df = pd.read_csv("/content/storage_nodes_12.csv")
files_df = pd.read_csv("/content/data_blocks_200.csv")

num_nodes = len(nodes_df)
num_files = len(files_df)

node_capacity = nodes_df["capacity_gb"].values
node_energy = nodes_df["energy_consumption_watts"].values
file_sizes = files_df["file_size_gb"].values

np.random.seed(42)
random.seed(42)

# =====================================================
# 2. FITNESS FUNCTION
# =====================================================
def calculate_fitness(assignment):
    load = np.zeros(num_nodes)

    for i, node in enumerate(assignment):
        load[node] += file_sizes[i]

    load_percent = load / node_capacity
    load_variance = np.var(load_percent)

    idle_power_ratio = 0.3
    energy_usage = 0
    for i in range(num_nodes):
        if load[i] > 0:
            utilization = load[i] / node_capacity[i]
            energy_usage += node_energy[i] * (idle_power_ratio + utilization)

    overload_penalty = np.sum(np.maximum(0, load_percent - 1.0) ** 2)
    underuse_penalty = np.sum(np.maximum(0, 0.3 - load_percent) ** 2)

    fitness = (
        10 * load_variance
        + 0. * energy_usage
        + 300 * overload_penalty
        + 30 * underuse_penalty
    )

    return fitness, load_percent, load_variance, energy_usage, overload_penalty, underuse_penalty

# =====================================================
# 3. GREEDY INITIALIZATION
# =====================================================
def greedy_assignment():
    assignment = np.zeros(num_files, dtype=int)
    load = np.zeros(num_nodes)

    for i in np.argsort(-file_sizes):
        score = (load / node_capacity) + 0.6 * node_energy
        node = np.argmin(score)
        assignment[i] = node
        load[node] += file_sizes[i]

    return assignment

# =====================================================
# 4. BASELINE SOLUTION
# =====================================================
baseline_assignment = np.random.randint(0, num_nodes, num_files)
baseline_results = calculate_fitness(baseline_assignment)

baseline_fitness = baseline_results[0]
baseline_load = baseline_results[1]
baseline_load_var = baseline_results[2]
baseline_energy = baseline_results[3]
baseline_overload = baseline_results[4]
baseline_underuse = baseline_results[5]

print(f"Baseline Fitness: {baseline_fitness:.6f}")

# =====================================================
# 5. PSO PARAMETERS
# =====================================================
num_particles = 50
iterations = 150
elite_count = 2

# =====================================================
# 6. INITIALIZE PARTICLES
# =====================================================
particles = []

for i in range(num_particles):
    if i < num_particles // 2:
        particles.append(greedy_assignment())
    else:
        particles.append(np.random.randint(0, num_nodes, num_files))

particles = np.array(particles)

pbest = particles.copy()
pbest_fitness = np.array([calculate_fitness(p)[0] for p in particles])

gbest_idx = np.argmin(pbest_fitness)
gbest = pbest[gbest_idx].copy()
gbest_fitness = pbest_fitness[gbest_idx]

fitness_history = []

# =====================================================
# 7. PSO OPTIMIZATION LOOP
# =====================================================
for it in range(iterations):

    mutation_rate = 0.5 * (1 - it / iterations)
    elite_idx = np.argsort(pbest_fitness)[:elite_count]

    for i in range(num_particles):

        if i in elite_idx:
            continue

        for f in range(num_files):
            r = random.random()
            if r < 0.35:
                particles[i][f] = pbest[i][f]
            elif r < 0.75:
                particles[i][f] = gbest[f]

        if random.random() < mutation_rate:
            load = np.zeros(num_nodes)
            for idx, n in enumerate(particles[i]):
                load[n] += file_sizes[idx]

            overloaded = np.argmax(load)
            underloaded = np.argmin(load)

            candidates = np.where(particles[i] == overloaded)[0]
            if len(candidates) > 0:
                heavy_files = candidates[np.argsort(-file_sizes[candidates])[:3]]
                for hf in heavy_files:
                    particles[i][hf] = underloaded

        fit = calculate_fitness(particles[i])[0]

        if fit < pbest_fitness[i]:
            pbest[i] = particles[i].copy()
            pbest_fitness[i] = fit

    gbest_idx = np.argmin(pbest_fitness)
    gbest = pbest[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    fitness_history.append(gbest_fitness)

    if (it + 1) % 10 == 0:
        print(f"Iteration {it+1} | Best Fitness: {gbest_fitness:.6f}")

# =====================================================
# 8. FINAL RESULTS
# =====================================================
opt_results = calculate_fitness(gbest)

optimized_fitness = opt_results[0]
optimized_load = opt_results[1]
opt_load_var = opt_results[2]
opt_energy = opt_results[3]
opt_overload = opt_results[4]
opt_underuse = opt_results[5]

gain = ((baseline_fitness - optimized_fitness) / baseline_fitness) * 100

print("\n--- FINAL RESULTS ---")
print(f"Optimized Fitness: {optimized_fitness:.6f}")
print(f"Optimization Gain: {gain:.2f}%")

# =====================================================
# 9. INDIVIDUAL OBJECTIVE REPORTING (NUMERIC ONLY)
# =====================================================
print("\n--- INDIVIDUAL OBJECTIVE COMPARISON ---")
print(f"Load Variance   | Baseline: {baseline_load_var:.6f} | Optimized: {opt_load_var:.6f}")
print(f"Energy Usage    | Baseline: {baseline_energy:.6f} | Optimized: {opt_energy:.6f}")
print(f"Overload Penalty| Baseline: {baseline_overload:.6f} | Optimized: {opt_overload:.6f}")
print(f"Underuse Penalty| Baseline: {baseline_underuse:.6f} | Optimized: {opt_underuse:.6f}")

# =====================================================
# 10. SUMMARY TABLE
# =====================================================
summary_df = pd.DataFrame({
    "Metric": [
        "Fitness",
        "Load Variance",
        "Energy Consumption",
        "Overload Penalty",
        "Underuse Penalty"
    ],
    "Baseline": [
        baseline_fitness,
        baseline_load_var,
        baseline_energy,
        baseline_overload,
        baseline_underuse
    ],
    "Optimized (PSO)": [
        optimized_fitness,
        opt_load_var,
        opt_energy,
        opt_overload,
        opt_underuse
    ]
})

print("\n--- FINAL SUMMARY TABLE ---")
print(summary_df)

# =====================================================
# 11. PLOTS (ONLY VALID ONES)
# =====================================================

# PSO Convergence
plt.figure()
plt.plot(fitness_history, color='red')
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.title("PSO Convergence")
plt.grid(True)
plt.show()

# Load Distribution (REAL PSO OUTPUT)
x = np.arange(num_nodes)
plt.figure()

plt.bar(
    x - 0.3,
    baseline_load * 100,
    width=0.3,
    label="Baseline",
    color="Red"
)

plt.bar(
    x,
    optimized_load * 100,
    width=0.3,
    label="PSO Optimized",
    color="green"
)

plt.axhline(100, linestyle="--", label="Capacity Limit")

plt.xlabel("Storage Nodes")
plt.ylabel("Load (%)")
plt.title("Load Distribution Comparison")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# 12. HOT vs COLD DATA HEATMAP (ANALYSIS ONLY)
# =====================================================
data_access_frequency = np.random.exponential(scale=1.0, size=num_files)
data_access_frequency /= np.max(data_access_frequency)

heatmap_matrix = np.zeros((num_nodes, num_files))
for file_idx in range(num_files):
    node = gbest[file_idx]
    heatmap_matrix[node, file_idx] = data_access_frequency[file_idx]

plt.figure(figsize=(12, 5))
plt.imshow(
    heatmap_matrix,
    aspect="auto",
    cmap="Reds"   # 🔥 Orange → Red colormap
)
plt.colorbar(label="Data Access Frequency (Cold → Hot)")
plt.xlabel("Data Blocks")
plt.ylabel("Storage Nodes")
plt.title("Hot vs Cold Data Placement Heatmap (PSO Optimized)")
plt.show()
