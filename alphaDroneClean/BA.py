# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import time

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
    episode_length_s = 2.0
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
        env_spacing=2.5, 
        replicate_physics=True,
    )


class DroneEnv(DirectRLEnv):
    cfg: DroneEnvCfg

    def __init__(self, cfg: DroneEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #list for collecting data
        self.data = [[] for _ in range(self.scene.cfg.num_envs)]
        self.drone.rotor_ids = self.scene.articulations["drone"].find_bodies("rotor_[1-4]")

        #self.drone_imu = Imu(self.drone.imu_cfg.replace(prim_path="/World/envs/env_.*/drone/base_link/IMU"))
        #self.scene.sensors["imu"] = self.drone_imu

        #self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.drone = Drone(self.scene.cfg.num_envs)
        #self.scene.articulations["drone"] = Articulation(self.drone.drone_cfg.replace(prim_path="/World/envs/env_.*/drone"))
        #self.scene.articulations["drone"] = self.scene.articulations["drone"]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=True)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()

        self.drone.controller.step(desired_states=self.drone.setpoint, states=self.scene.articulations["drone"].data.body_com_state_w, dt=self.cfg.sim.dt)

        #print(self.scene.sensors["imu"].data.lin_acc_b)
        #print(self.drone.controller.ve_thrust)

    def _apply_action(self):
        self.scene.articulations["drone"].set_external_force_and_torque(
                self.drone.controller.ve_thrust, 
                self.drone.controller.ve_moment,
                body_ids=self.drone.rotor_ids[0], 
                is_global=False
                )

    def _get_observations(self) -> dict:
        #self.observedPosition,  self.observedQuat, self.observedVelocity, self.observedAngularVelocity = self.scene.articulations["drone"].data.body_com_state_w
        #observations = {
        #        "pos": self.observedPosition,
        #        "vel": self.observedVelocity,
        #        "quat": self.observedQuat,
        #        "angVel": self.observedAngVelocity,
        #        "action": self.drone.thrust,
        #        "setpoint": self.drone.setpoint
        #        }
        observations = {}

        for i in range(self.scene.cfg.num_envs):
            flatten = torch.cat([
                self.scene.articulations["drone"].data.body_com_pos_b[i][0],
                self.scene.articulations["drone"].data.body_com_quat_b[i][0],
                self.scene.articulations["drone"].data.body_lin_vel_w[i][0],
                self.scene.articulations["drone"].data.body_ang_vel_w[i][0],
                self.scene.sensors["imu"].data.lin_acc_b[i],
                self.scene.sensors["imu"].data.ang_vel_b[i],
                self.drone.thrust[i],
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
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        #died = torch.logical_or(self.scene.articulations["drone"].data.root_pos_w[:, 2] < 0.0, self.scene.articulations["drone"].data.root_pos_w[:, 2] > 20.0)
        died = torch.zeros_like(time_out, dtype=torch.bool)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.scene.articulations["drone"]._ALL_INDICES

        joint_pos = self.scene.articulations["drone"].data.default_joint_pos[env_ids]
        joint_vel = self.scene.articulations["drone"].data.default_joint_vel[env_ids]
        default_root_state = self.scene.articulations["drone"].data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.scene.articulations["drone"].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.scene.articulations["drone"].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.scene.articulations["drone"].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.episode_length_buf[env_ids] = 0
    
        for i in env_ids:
            if not self.data[i]:     
                continue

            timestamp = int(time.time())
            t = torch.stack(self.data[i]).to(torch.float32).contiguous()

            filename = f"data/drone_{i}_{timestamp}.bin"

            with open(filename, "wb") as f:
                f.write(t.cpu().numpy().tobytes())

            self.data[i].clear()

