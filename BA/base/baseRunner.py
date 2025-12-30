import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from baseDrone import Drone
from baseEnvironment import DroneEnvironment

class QuadcopterEnv(DirectRLEnv):
    cfg : DroneEnvironment
    drone : Drone

    def __init__(self, cfg: DroneEnvironment, render_mode: str | None = None, **kwargs):
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
        self._drone = Articulation(self.drone.articulation_cfg)
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


