# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

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

drone_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(usd_path="/home/azapata/JoeysStuffs/alphaDrone-Isaac/URDF_template/test3.usda"),
        actuators={"rotors": ImplicitActuatorCfg(joint_names_expr=["rotor_[1-4]_joint"], damping=None, stiffness=None)},
        init_state=ArticulationCfg.InitialStateCfg(pos=[0.0, 0.0, 0.2])
        )   

class QuadcopterEnvWindow(BaseEnvWindow):

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    episode_length_s = 2.0
    decimation = 1
    action_space = 4
    observation_space = 6
    state_space = 0
    debug_vis = False

    ui_window_class_type = QuadcopterEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2, 
        env_spacing=2.5, 
        replicate_physics=True
    )

    drone = drone_cfg.replace(prim_path="/World/envs/env_.*/drone")

    reward_alive_scale = 1.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx = self._drone.find_joints(name_keys=["rotor_[1-4]_joint"])

        self._thrust = torch.zeros(self.num_envs, len(self.dof_idx[0]), 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, len(self.dof_idx[0]), 3, device=self.device)

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_pos_w[:, 2] = 4.0

        self._body_id = self._drone.find_bodies("rotor_[1-4]")
        self._drone_mass = self._drone.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._drone_weight = (self._drone_mass * self._gravity_magnitude).item()

        #self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._drone = Articulation(self.cfg.drone)
        self.scene.articulations["drone"] = self._drone

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

        self._thrust[:, :, 1] = self._actions[:, :]
        self._moment = self._thrust * 10.0

        self._moment[:, 1, 1] *= -1
        self._moment[:, 3, 1] *= -1

        print(self._thrust)
        print("\n")

    def _apply_action(self):
        self._drone.set_external_force_and_torque(
                self._thrust, 
                self._moment,
                body_ids=self._body_id[0], 
                is_global=False
                )

    def _get_observations(self) -> dict:
        self.observedPosition = self._drone.data.root_link_pos_w
        observations = {"policy": self.observedPosition}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._drone.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
                "distance_to_goal": distance_to_goal_mapped * self.step_dt,
                }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._drone.data.root_pos_w[:, 2] < 0.1, self._drone.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._drone._ALL_INDICES

        joint_pos = self._drone.data.default_joint_pos[env_ids]
        joint_vel = self._drone.data.default_joint_vel[env_ids]
        default_root_state = self._drone.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._drone.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._drone.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._drone.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


