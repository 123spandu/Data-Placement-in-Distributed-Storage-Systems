# A PSO-Based Framework for Energy-Aware Load-Balanced Data Placement in Distributed Storage Systems

## 📌 Overview
This project presents a **Particle Swarm Optimization (PSO)–based framework** for optimizing **data placement in distributed storage systems**. The proposed approach focuses on achieving **energy efficiency**, **load balancing**, and **effective resource utilization** using a **single- fitness formulation** that evaluates multiple system-level metrics.objective

The framework also incorporates **content-awareness** through hot and cold data analysis and demonstrates **scalability** using datasets of varying sizes.

---

## 🎯 Objectives
- Minimize load imbalance across storage nodes  
- Reduce overall energy consumption  
- Prevent storage node overload  
- Reduce underutilization of resources  
- Analyze data access patterns using hot and cold data classification  
- Evaluate scalability across multiple dataset sizes  

---

## 🧠 Methodology
- **Optimization Algorithm:** Particle Swarm Optimization (PSO)
- **Particle Representation:** Mapping of data blocks to storage nodes
- **Fitness Function Components:**
  - Load Variance
  - Energy Consumption
  - Overload Penalty
  - Underutilization Penalty
- **Enhancements:**
  - Greedy + Random particle initialization
  - Discrete PSO update strategy
  - Elite particle preservation
  - Mutation strategy for exploration

---

## 🏗️ System Architecture
1. Input data blocks and storage node configuration  
2. Particle initialization using greedy and random strategies  
3. Fitness evaluation  
4. PSO-based optimization loop  
5. Mutation and elite selection  
6. Convergence checking  
7. Output of optimized data placement (gbest)  

---

## 🧪 Experimental Setup
- **Number of Iterations:** 150 (fixed for all experiments)
- **Experimental Scenarios:**
  - 8 storage nodes with 100 data blocks
  - 12 storage nodes with 200 data blocks
  - 20 storage nodes with 400 data blocks
- **Evaluation Metrics:**
  - Fitness value
  - Load variance
  - Energy consumption
  - Overload penalty
  - Underutilization penalty

---

## 📊 Results Summary
- Significant reduction in fitness values across all experiments  
- Load variance reduced by up to **82.5%**
- Consistent reduction in energy consumption
- Overload penalties eliminated or significantly reduced
- Demonstrated scalability with increasing data size and system complexity

---

## 🚀 Key Contributions
- Single-objective PSO with multi-metric evaluation  
- Content-aware data placement using hot and cold data analysis  
- Discrete PSO with mutation and elite selection  
- Scalable and energy-efficient optimization framework  

---

## 🔮 Future Work
- Extension to dynamic and real-time workloads  
- Integration of fault tolerance and data replication  
- Hybrid optimization using PSO with GA or ACO  
- Deployment on cloud and edge storage systems  
- Incorporation of security-aware and reliability-focused objectives  

---

## 🛠️ Technologies Used
- Python  
- NumPy  
- Matplotlib  
- LaTeX (IEEE Conference Format)  

---

## 👩‍💻 Author
**Spandana N**  
Master of Computer Applications  
RV College of Engineering, Bengaluru  

  
