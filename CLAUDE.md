# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Isaac Lab/Isaac Sim drone simulation for generating flight data used to train IMU-based state estimators (thesis project). The simulation collects normalized 23-dimensional observations at 200 Hz and writes them to Zarr archives on disk.

## Running the Simulation

All entry points must go through Isaac Lab's `AppLauncher`, which initializes the Omniverse runtime before any Isaac imports:

```bash
# From alphaDroneClean/
python agent.py --task run --num_envs <N> [--headless]
```

`--headless` is required on machines without a display. `--num_envs` controls parallel environment count (default 2).

The simulation stops automatically once `max_iter=10000` episodes are collected. Data is written to `/media/egghead/Scratch/joey/simulation_data/`.

## Running Tests

```bash
# Standalone LQR controller test (no Isaac required)
cd alphaDroneClean/include && python ../test/controller_test.py

# PyBullet URDF sanity check
cd alphaDroneClean/fly_boy && python fly_boy_test.py
```

## Architecture

### Entry Flow

`agent.py` → parses CLI args and launches `AppLauncher` → imports `task.py` which registers the gym env → `gym.make("run")` instantiates `DroneEnv` from `BA.py`.

### Core Files (`alphaDroneClean/`)

| File | Role |
|------|------|
| `BA.py` | `DroneEnv` (Isaac Lab `DirectRLEnv` subclass), `DroneEnvCfg`, `Drone` asset config |
| `agent.py` | CLI entry point; must be the first file run (initializes Omniverse) |
| `task.py` | Gym registration; imported after AppLauncher |
| `include/Kxontroller.py` | Batched LQR controller + `Kilter` Kalman filter, gains loaded from `controller.yaml` |
| `include/QuickActionPrim.py` | `ActionPrimitive` manager + primitive types: `HoldPosition`, `RandomWalk`, `CubicSpline`, `RandomSphereOffset` |
| `include/QuickMavMulti.py` | MAVLink SITL bridge (TCP 4560+, UDP 14580+ per env) |
| `include/DataWriter.py` | Zarr episode writer |
| `include/Planner.py` | Vectorized Bézier curve planner |
| `include/controller.yaml` | Precomputed LQR/Kalman matrices (A, B, K_LQR, P_LQR, etc.) |

### Two Operating Modes

`DroneEnv.sitl` (bool, set in `__init__`):

- **`sitl = False`** (default): Pure sim. The `Kxontroller` LQR runs inside Python and drives rotor forces directly. Gaussian thrust/moment noise is added per-step using a per-episode `noise_scale` sampled uniformly in `[1e-5, 1e-1]`. This is the data collection mode.

- **`sitl = True`**: PX4 SITL in hardware-in-the-loop. IMU and position targets are sent over MAVLink; actuator commands come back via `HIL_ACTUATOR_CONTROLS`. A grace period (`grace_steps`) gates arming after each reset to let PX4 reinitialize.

### Simulation Loop Timing

- Physics runs at **800 Hz** (`dt = 1/800`)
- Controller and data collection run at **200 Hz** (`steps_per_sample = 4`)
- `_pre_physics_step` fires every tick but only acts every `steps_per_sample` ticks

### Coordinate Frame Convention

Isaac Lab uses NWU (x-forward, y-left, z-up). PX4/MAVLink expects NED/FRD. The conversion applied before sending MAVLink messages or storing observations:
- `pos[:, 1:]`, `lin_vel[:, 1:]`, `ang_vel[:, 1:]`, `acc[:, 1:]` → multiply by `-1`
- Quaternion: negate `y` and `z` components → `quat_frd`

### Observation Vector (23-dim, normalized)

```
pos/3 (3) | lin_vel/0.8 (3) | quat_frd (4) | ang_vel/0.5 (3) | acc/25 (3) | thrust/9.81 (4) | setpoint/3 (3)
```

### Domain Randomization

On every episode reset (`_reset_idx`):
- Mass scaled uniformly in `[0.98, 1.02]×`
- CoM offset randomized ±1 mm (x/y), ±1 cm (z)
- `noise_scale` resampled uniformly in `[1e-5, 1e-1]`

### Drone Asset

USDA/URDF in `alphaDroneClean/fly_boy/`. The articulation has 4 rotor joints (`rotor_[1-4]_joint`). Forces are applied as external force+torque on each rotor body: thrust along local Z, moment = `thrust × 0.09` with rotors 1 and 3 counter-rotating (moment Z negated).

### ActionPrimitive

Manages per-environment trajectory primitives that switch randomly after `min_duration`–`max_duration` steps. Each primitive implements `reset(ids, start_pos, generator)` and `step(ids, tau, out)`. Setpoints are clamped to `[(-2.5,-2.5,0.8), (2.5,2.5,2.5)]` with reflection at bounds.
