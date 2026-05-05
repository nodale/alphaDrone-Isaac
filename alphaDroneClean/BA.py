# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import time
import zarr

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ImuCfg, Imu

from pxr import Gf
from Kxontroller import Kxontroller

class Drone():
    drone_cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/drone",
            spawn=sim_utils.UsdFileCfg(usd_path="drone/test9.usda"),
            actuators={"rotors": ImplicitActuatorCfg(joint_names_expr=["rotor_[1-4]_joint"], damping=None, stiffness=None)},
            init_state=ArticulationCfg.InitialStateCfg(pos=[0.0, 0.0, 0.2])
            )   

    imu_cfg = ImuCfg(
            prim_path="{ENV_REGEX_NS}/drone/base_link",
            update_period=0.0,
            history_length=0,
            debug_vis=False
            )

    def __init__(self, num_envs, device="cuda"):
        self.thrust = torch.zeros(num_envs, 4, device=device)
        self.moment = torch.zeros(num_envs, 4, device=device)

        self.controller = Kxontroller(num_envs=num_envs)
        self.rotor_ids = torch.zeros(num_envs, 4, device=device)

        self.setpoint = torch.zeros(num_envs, 3, device=device)

        for sp in self.setpoint:
            sp[:] = torch.tensor([0.0, 0.0, 2.0], device=device)

class DataWriter():
    def __init__(self, path="dataset/all_data.zarr/", num_points=10, num_batch=10, n_dim=12):
        self.path = path
        self.num_points = int(num_points)
        self.num_batch = int(num_batch)
        self.n_dim = int(n_dim)

        self.batch_idx = 0 

        if self.path is not None:
            self.store = zarr.storage.LocalStore(self.path)
            self.root = zarr.group(store=self.store, overwrite=True)
            self.data = self.root.create_group('data')

            self.arrays = {
                i: self.data.create_array(
                    name=f"dim_{i}",
                    shape=(self.num_batch, self.num_points),
                    chunks=(1, self.num_points), 
                    dtype="f4",
                    overwrite=True
                )
                for i in range(self.n_dim)
            }

    def __del__(self):
        self.store.close()

    def write_episode(self, data):
        # data: (num_points, n_dim)

        if self.batch_idx >= self.num_batch:
            print(" DataWriter full, skipping write")
            return

        if data.shape[0] != self.num_points:
            raise ValueError(f"Expected {self.num_points} timesteps, got {data.shape[0]}")

        for i, arr in self.arrays.items():
            arr[self.batch_idx, :] = data[:, i]

        self.batch_idx += 1
        
class DroneEnvWindow(BaseEnvWindow):
    def __init__(self, env: DroneEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)




@configclass
class DroneSceneCfg(InteractiveSceneCfg):
    imu = Drone.imu_cfg
    drone = Drone.drone_cfg

@configclass
class DroneEnvCfg(DirectRLEnvCfg):
    episode_length_s = 1.0
    decimation = 1
    action_space = 4
    observation_space = 6
    state_space = 0
    debug_vis = False
    dt = 1.0/200.0
    
    ui_window_class_type = DroneEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = DroneSceneCfg(
        num_envs=2, 
        env_spacing=2.0, 
        replicate_physics=True,
    )


class DroneEnv(DirectRLEnv):
    cfg: DroneEnvCfg

    def __init__(self, cfg: DroneEnvCfg, render_mode: str | None = None, **kwargs):
        #super().__init__(cfg, render_mode, **kwargs)
        super().__init__(cfg, render_mode="None", **kwargs)

        #list for collecting data
        self.data = [[] for _ in range(self.scene.cfg.num_envs)]
        self.drone.rotor_ids = self.scene.articulations["drone"].find_bodies("rotor_[1-4]")

        #data writer
        self.max_iter = 100000
        self.seq_len = self.cfg.episode_length_s / self.cfg.sim.dt
        self.writer = DataWriter(num_batch=self.max_iter, num_points=self.seq_len, n_dim=12)
        self.t1 = 1.0
        self.t0 = 0.0

    def _setup_scene(self):
        self.drone = Drone(self.scene.cfg.num_envs)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=True)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        #light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        #light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()

        self.drone.controller.step(desired_states=self.drone.setpoint, states=self.scene.articulations["drone"].data.root_com_state_w, dt=self.cfg.sim.dt)

    def _apply_action(self):
        self.scene.articulations["drone"].set_external_force_and_torque(
                self.drone.controller.ve_thrust, 
                self.drone.controller.ve_moment,
                body_ids=self.drone.rotor_ids[0], 
                is_global=False
                )

    def _get_observations(self) -> dict:
        observations = {}

        for i in range(self.scene.cfg.num_envs):
            #environment frame to local body frame
            quat = self.scene.articulations["drone"].data.root_com_quat_w[i],
            _pre_rot = Gf.Quatd(
                    float(quat[0][0]),
                Gf.Vec3d(
                    float(quat[0][1]),
                    float(quat[0][2]),
                    float(quat[0][3])
                )
            )
            _gf_rot = Gf.Rotation(_pre_rot) 
            _rot = _gf_rot.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
            rot = torch.tensor([_rot[0] * 0.0174532925, _rot[1] * 0.0174532925, _rot[2] * 0.0174532925], dtype=torch.float32, device="cuda")

            roll, pitch, yaw = rot
            cr = torch.cos(roll)
            sr = torch.sin(roll)
            cp = torch.cos(pitch)
            sp = torch.sin(pitch)
            cy = torch.cos(yaw)
            sy = torch.sin(yaw)

            R = torch.tensor([
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp,   cp*sr,            cp*cr]
            ], dtype=torch.float32, device="cuda")

            local_lin_vel = R.T @ self.scene.articulations["drone"].data.root_com_lin_vel_b[i]
            local_ang_vel = R.T @ self.scene.articulations["drone"].data.root_com_ang_vel_b[i]
            
            flatten = torch.cat([
                self.scene.articulations["drone"].data.root_com_pos_w[i],
                local_lin_vel,
                rot,
                local_ang_vel,
                self.scene.sensors["imu"].data.lin_acc_b[i],
                self.scene.sensors["imu"].data.ang_vel_b[i],
                self.drone.controller.thrust[i],
                self.drone.setpoint[i]
                ])

            self.data[i].append(flatten.clone())

        return observations

    def _get_rewards(self) -> torch.Tensor:
        distance_to_goal = torch.linalg.norm(self.scene.articulations["drone"].data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
                "distance_to_goal": distance_to_goal_mapped * self.step_dt,
                }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        #died = torch.logical_or(self.scene.articulations["drone"].data.root_pos_w[:, 2] < 0.0, self.scene.articulations["drone"].data.root_pos_w[:, 2] > 20.0)
        died = torch.zeros_like(time_out, dtype=torch.bool)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.scene.articulations["drone"]._ALL_INDICES

        # clone defaults so we don’t overwrite internal buffers
        joint_pos = self.scene.articulations["drone"].data.default_joint_pos[env_ids].clone()
        joint_vel = self.scene.articulations["drone"].data.default_joint_vel[env_ids].clone()
        root_state = self.scene.articulations["drone"].data.default_root_state[env_ids].clone()

        # --- POSITION RANDOMIZATION ---
        pos_noise = torch.empty((len(env_ids), 3), device=self.device).uniform_(-0.5, 0.5)
        root_state[:, :3] += pos_noise

        # keep above ground
        root_state[:, 2] = torch.clamp(root_state[:, 2], min=0.2)

        # add env origins
        root_state[:, :3] += self._terrain.env_origins[env_ids]

        # --- ORIENTATION RANDOMIZATION (yaw only for stability) ---
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi/20.0, torch.pi/20.0)

        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        # quaternion (x, y, z, w) → Isaac uses (w, x, y, z)
        root_state[:, 3:7] = torch.stack([
            cy,                         # w
            torch.zeros_like(cy),       # x
            torch.zeros_like(cy),       # y
            sy                          # z
        ], dim=1)

        # --- LINEAR VELOCITY RANDOMIZATION ---
        lin_vel_noise = torch.empty((len(env_ids), 3), device=self.device).uniform_(-2.0, 2.0)
        root_state[:, 7:10] = lin_vel_noise

        # --- ANGULAR VELOCITY RANDOMIZATION ---
        ang_vel_noise = torch.empty((len(env_ids), 3), device=self.device).uniform_(-0.1, 0.1)
        root_state[:, 10:13] = ang_vel_noise

        # --- JOINT RANDOMIZATION (rotors) ---
        joint_vel_noise = torch.empty_like(joint_vel).uniform_(-0.1, 0.1)
        joint_vel += joint_vel_noise

        # (optional) small position noise if joints have meaningful angles
        joint_pos_noise = torch.empty_like(joint_pos).uniform_(-0.1, 0.1)
        joint_pos += joint_pos_noise

        # --- WRITE TO SIM ---
        self.scene.articulations["drone"].write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.scene.articulations["drone"].write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.scene.articulations["drone"].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.episode_length_buf[env_ids] = 0

        # random offset (relative target)
        low = torch.tensor([-1.0, -1.0, 0.5], device=self.device)
        high = torch.tensor([1.0, 1.0, 3.0], device=self.device)

        rand = torch.rand((len(env_ids), 3), device=self.device)
        offset = low + (high - low) * rand

        # base position = environment origin (spawn reference)
        base_pos = self._terrain.env_origins[env_ids]

        # final setpoint = relative to spawn
        self.drone.setpoint[env_ids] = base_pos + offset

        #printing the data, so we store each environment intoa buffer, then print em when it is reset
        for i in env_ids:
            if not self.data[i]:
                continue

            t = torch.stack(self.data[i])  # (T, dim)

            if t.shape[0] != self.seq_len:
                continue  # or pad

            t = t.cpu().numpy()

            self.writer.write_episode(t)

            self.data[i].clear()

        print("progress : ", self.writer.batch_idx, " out of ", self.writer.num_batch)
        if self.writer.batch_idx >= self.writer.num_batch:
            print("Dataset complete. Stopping simulation...")

            self.t0=self.t1
            self.t1=time.perf_counter()

            print("time : ", self.t1 - self.t0)

            self.writer.store.close()
            raise SystemExit
