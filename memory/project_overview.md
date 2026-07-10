---
name: project-overview
description: High-level purpose and architecture of alphaDrone-Isaac thesis project
metadata:
  type: project
---

Isaac Lab/Isaac Sim drone simulation for generating IMU flight data for thesis on state estimation.

**Why:** Thesis project — synthetic data collection at scale (10,000 episodes) to train neural state estimators.

**How to apply:** Treat data collection throughput and simulation fidelity as the primary constraints when suggesting changes. The `sitl=False` path is the active one; SITL mode is experimental.

Key facts:
- Entry: `python agent.py --task run --num_envs N [--headless]` from `alphaDroneClean/`
- Data written to `/media/egghead/Scratch/joey/simulation_data/` (zarr format)
- 800 Hz physics, 200 Hz data collection, 23-dim normalized obs vector
- Two modes: pure-sim LQR (default) and PX4 SITL over MAVLink
- Coordinate frame: Isaac NWU → PX4 NED/FRD by negating y/z components
- Domain randomization: mass ±2%, CoM offset, noise_scale per episode
