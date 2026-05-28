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
from include.Kxontroller import Kxontroller
from include.Planner import Planner

class Drone():
    drone_cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/drone",
            spawn=sim_utils.UsdFileCfg(usd_path="fly_boy/fly_boy.usda"),
            actuators={"rotors": ImplicitActuatorCfg(joint_names_expr=["rotor_[1-4]_joint"], damping=None, stiffness=None)},
            init_state=ArticulationCfg.InitialStateCfg(pos=[0.0, 0.0, 0.2])
            )   

    imu_cfg = ImuCfg(
            prim_path="{ENV_REGEX_NS}/drone/base_link",
            update_period=1.0/200.0,
            history_length=2,
            debug_vis=False
            )

    def __init__(self, num_envs, device="cuda"):
        self.thrust = torch.zeros(num_envs, 4, device=device)
        self.moment = torch.zeros(num_envs, 4, device=device)

        self.controller = Kxontroller(num_envs=num_envs)
        self.rotor_ids = torch.zeros(num_envs, 4, device=device)

        self.setpoint = torch.zeros(num_envs, 3, device=device)

class DataWriter():
    def __init__(self, path="/home/joey/Thesis/data/patient_one_data.zarr/", num_batch=10, seq_len=10, n_dim=25):
        self.path = path
        self.num_batch = int(num_batch)
        self.seq_len = int(seq_len)
        self.n_dim = int(n_dim)

        self.batch_idx = 0 

        self.store = zarr.storage.LocalStore(self.path)
        self.root = zarr.group(store=self.store, overwrite=True)

        self.data = self.root.create_array(
            name="episodes",
            shape=(self.num_batch, self.seq_len, self.n_dim),
            chunks=(1, self.seq_len, self.n_dim), 
            dtype="f8",
            overwrite=True
        )

    def __del__(self):
        self.store.close()

    def write_episode(self, data):
        #(seq_len, n_dim)
        self.data[self.batch_idx, :, :] = data
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
    episode_length_s = 8.0
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
        env_spacing=1e-3, 
        replicate_physics=True,
    )


class DroneEnv(DirectRLEnv):
    cfg: DroneEnvCfg

    def __init__(self, cfg: DroneEnvCfg, render_mode: str | None = None, **kwargs):
        #super().__init__(cfg, render_mode=None, **kwargs)
        super().__init__(cfg, render_mode, **kwargs)

        #data writer
        self.n_dim = 27
        self.max_iter = 500
        self.seq_len = int(self.cfg.episode_length_s / self.cfg.sim.dt)
        self.writer = DataWriter(num_batch=self.max_iter, seq_len=self.seq_len, n_dim=self.n_dim)
        self.t1 = 1.0
        self.t0 = 0.0

        #list for collecting data
        self.data = torch.zeros(self.scene.num_envs, self.seq_len, self.n_dim, device="cuda")
        self.step_idx = torch.zeros(self.scene.num_envs, dtype=torch.long, device="cpu")
        self.drone.rotor_ids = self.scene.articulations["drone"].find_bodies("rotor_[1-4]")
        self.loop_counter = 1
        self.sampling_freq = 200.0

        #mission planner
        self.bezier = Planner.random(
                M=self.scene.num_envs, 
                seed=0,
                offset=self.scene.articulations["drone"].data.root_com_pos_w,
                device="cuda")

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
        self.loop_counter += 1
        env_timestamp = self.episode_length_buf.reshape(self.episode_length_buf.shape[0], 1) * self.cfg.sim.dt

        if self.loop_counter % (self.cfg.sim.dt * self.sampling_freq) == 0:
            _sp = self.bezier.step(self.scene.articulations["drone"].data.root_com_pos_w, dt=self.cfg.sim.dt ,mode="pos")
            self.drone.setpoint = _sp

            drone_data = self.scene.articulations["drone"].data
            imu_data = self.scene.sensors["imu"].data

            acc =  imu_data.lin_acc_b.detach().clone()
            gyro =  imu_data.ang_vel_b.detach().clone()
            pos = drone_data.root_com_pos_w.detach().clone()                      # (N, 3)
            quat = drone_data.root_com_quat_w.detach().clone()                    # (N, 4) [w, x, y, z]
            lin_vel = drone_data.root_com_lin_vel_b.detach().clone()              # (N, 3)
            ang_vel = drone_data.root_com_ang_vel_b.detach().clone()              # (N, 3)
            setpoint = _sp.detach().clone()

            w, x, y, z = quat.unbind(dim=1)
            quat_frd = torch.stack([
                w,
                x,
                -y,
                -z
            ], dim=1)

            #transforming the data to PX4 orientation
            pos[:, 1:] *= -1.0
            lin_vel[:, 1:] *= -1.0
            ang_vel[:, 1:] *= -1.0
            acc[:, 1:] *= -1.0
            gyro[:, 1:] *= -1.0
            setpoint[:, 1:] *= -1.0            

            obs = torch.cat([
                pos,
                lin_vel,
                quat_frd,
                ang_vel,
                acc,
                gyro,
                self.drone.controller.thrust,
                setpoint,
                env_timestamp,
            ], dim=1)  # (N, dim)

            valid = self.step_idx < self.seq_len  # (N,)
            idx = self.step_idx[valid]

            self.data[valid, idx] = obs[valid]
            self.step_idx[valid] += 1

            #setpoint = torch.tensor([[
            #    0.0, 0.0, -1.0]], device="cuda")
            self.drone.controller.step(desired_pos=setpoint, states=obs[:, :13], dt=1.0/self.sampling_freq)
            #self.scene.articulations["drone"].data.root_com_state_w

            #print(self.drone.controller.ve_thrust[0, :], setpoint[0, :], obs[:, :13])

    def _apply_action(self):
        self.scene.articulations["drone"].set_external_force_and_torque(
                self.drone.controller.ve_thrust, 
                self.drone.controller.ve_moment,
                body_ids=self.drone.rotor_ids[0], 
                is_global=False
                )

    def _get_observations(self) -> dict:
        observations = {}

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
        root_state[:, 2] = torch.clamp(root_state[:, 2], min=0.1)

        # add env origins
        root_state[:, :3] += self._terrain.env_origins[env_ids]

        # --- ORIENTATION RANDOMIZATION (yaw only for stability) ---
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi/80.0, torch.pi/80.0)

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
        lin_vel_noise = torch.empty((len(env_ids), 3), device=self.device).uniform_(-0.01, 0.01)
        root_state[:, 7:10] = lin_vel_noise

        # --- ANGULAR VELOCITY RANDOMIZATION ---
        ang_vel_noise = torch.empty((len(env_ids), 3), device=self.device).uniform_(-0.001, 0.001)
        root_state[:, 10:13] = ang_vel_noise

        # --- JOINT RANDOMIZATION (rotors) ---
        joint_vel_noise = torch.empty_like(joint_vel).uniform_(-0.02, 0.02)
        joint_vel += joint_vel_noise

        # (optional) small position noise if joints have meaningful angles
        joint_pos_noise = torch.empty_like(joint_pos).uniform_(-0.02, 0.02)
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
        self.bezier = Planner.random(
                M=self.scene.num_envs, 
                seed=0,
                offset=self.scene.articulations["drone"].data.root_com_pos_w,
                device="cuda")

        for i in env_ids:
            if self.step_idx[i] < self.seq_len:
                continue  

            t = self.data[i].detach().to("cpu", non_blocking=True).numpy()
            self.writer.write_episode(t)

            self.data[i].zero_()
            self.step_idx[i] = 0

        self.t0=self.t1
        self.t1=time.perf_counter()

        print("time : ", self.t1 - self.t0)

        print("progress : ", self.writer.batch_idx, " out of ", self.writer.num_batch)
        if self.writer.batch_idx >= self.writer.num_batch:
            print("Dataset complete. Stopping simulation...")


            self.writer.store.close()
            raise SystemExit
