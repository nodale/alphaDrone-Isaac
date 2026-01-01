import torch as torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from baseDrone import Drone
from baseEnvironment import DroneEnvironment

class DroneRunner(DirectRLEnv):
    cfg : DroneEnvironment
    drone : Drone

    def __init__(self, cfg: DroneEnvironment, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.drone = Drone(self.cfg.num_envs)
        self.scene.articulations["drone"] = self.drone.articulation
        self.drone.init()

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

        self.drone.thrust[:, :, 1] = self._actions[:, :]
        self.drone.moment = self._thrust * 10.0

        self.drone.moment[:, 1, 1] *= -1
        self.drone.moment[:, 3, 1] *= -1

    def _apply_action(self):
        self.drone.articulation.set_external_force_and_torque(
                self.drone.thrust, 
                self.drone.moment,
                body_ids=self.drone.rotor_id[0], 
                is_global=False
                )

    def _get_observations(self) -> dict:
        self.observedPosition = self.drone.articulation.data.body_link_pos_w
        observations = {"policy": self.observedPosition}
        return observations

    #def _get_rewards(self) -> torch.Tensor:
    #    distance_to_goal = torch.linalg.norm(self._desired_pos_w - self.drone.articulation.data.body_pos_w, dim=1)
    #    distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
    #    rewards = {
    #            "distance_to_goal": distance_to_goal_mapped * self.step_dt,
    #            }
    #    reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
    #    return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self.drone.articulation.data.body_pos_w[:, 2] < 0.1, self.drone.articulation.data.body_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.drone.articulation._ALL_INDICES

        joint_pos = self.drone.articulation.data.default_joint_pos[env_ids]
        joint_vel = self.drone.articulation.data.default_joint_vel[env_ids]
        default_root_state = self.drone.articulation.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.drone.articulation.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.drone.articulation.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.drone.articulation.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


