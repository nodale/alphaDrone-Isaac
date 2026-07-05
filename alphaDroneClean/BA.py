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
from isaaclab.managers import EventManager
from isaaclab.managers import EventTermCfg

from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs import DirectRLEnvCfg

from pxr import Gf
from include.Kxontroller import Kxontroller
from include.Planner import Planner
from include.QuickActionPrim import ActionPrimitive, HoldPosition, RandomWalk, CubicSpline, RandomSphereOffset
from include.DataWriter import DataWriter
from include.QuickMavMulti import QuickMavMulti

class Drone():
    drone_cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/drone",
            spawn=sim_utils.UsdFileCfg(usd_path="fly_boy/fly_boy.usda"),
            actuators={"rotors": ImplicitActuatorCfg(joint_names_expr=["rotor_[1-4]_joint"], damping=None, stiffness=None)},
            init_state=ArticulationCfg.InitialStateCfg(pos=[0.0, 0.0, 0.185])
            )   

    imu_cfg = ImuCfg(
            prim_path="{ENV_REGEX_NS}/drone/base_link",
            update_period=1.0/800.0,
            history_length=2,
            debug_vis=False
            )

    def __init__(self, num_envs, device="cuda"):
        #for esc delay
        self.esc_delay = 1.0/400.0

        self.thrust = torch.zeros(num_envs, 4, device=device)
        self.moment = torch.zeros(num_envs, 4, device=device)

        self.thrust_noise = torch.zeros(num_envs, 4, 3, device=device)
        self.moment_noise = torch.zeros(num_envs, 4, 3, device=device)

        self.controller = Kxontroller(num_envs=num_envs, esc_delay=self.esc_delay)
        self.rotor_ids = torch.zeros(num_envs, 4, device=device)

        self.setpoint = torch.zeros(num_envs, 3, device=device)


class DroneEnvWindow(BaseEnvWindow):
    def __init__(self, env: DroneEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)



@configclass
class MassRandomisationCfg:
    randomize_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "drone",
                body_names="base_link",
            ),
            "mass_distribution_params": (0.98, 1.02),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "drone",
                body_names="base_link",
            ),
            "com_range": {
                "x": (-0.001, 0.001),
                "y": (-0.001, 0.001),
                "z": (-0.01, 0.01),
            },
        },
    )


@configclass
class DroneSceneCfg(InteractiveSceneCfg):
    imu = Drone.imu_cfg
    drone = Drone.drone_cfg

@configclass
class DroneEnvCfg(DirectRLEnvCfg):
    episode_length_s = 5.0
    decimation = 1
    action_space = 4
    observation_space = 6
    state_space = 0
    debug_vis = False
    dt = 1.0/800.0
    seed = 100
    
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
        env_spacing=1e-5, 
        replicate_physics=True,
    )

    events = MassRandomisationCfg()


class DroneEnv(DirectRLEnv):
    cfg: DroneEnvCfg

    def __init__(self, cfg: DroneEnvCfg, render_mode: str | None = None, **kwargs):
        #super().__init__(cfg, render_mode=None, **kwargs)
        super().__init__(cfg, render_mode, **kwargs)

        #data writer
        self.n_dim = 23
        self.max_iter = 1000
        self.sampling_freq = 200.0
        self.steps_per_sample = int(round(1.0 / (self.cfg.sim.dt * self.sampling_freq)))
        self.seq_len = int((self.cfg.episode_length_s/self.steps_per_sample) / self.cfg.sim.dt)
        self.writer = DataWriter(num_batch=self.max_iter, seq_len=self.seq_len, n_dim=self.n_dim)
        self.t1 = 1.0
        self.t0 = 0.0

        #list for collecting data
        self.data = torch.zeros(self.scene.num_envs, self.seq_len, self.n_dim, device="cuda")
        self.step_idx = torch.zeros(self.scene.num_envs, dtype=torch.long, device="cpu")
        self.drone.rotor_ids = self.scene.articulations["drone"].find_bodies("rotor_[1-4]")
        self.loop_counter = 1

        #for SITL mavlink com
        self.sitl = True
        if self.sitl:
            self.mav = QuickMavMulti(
                    num_envs=self.scene.num_envs,
                    tcp_base=4560,
                    udp_base=14580)
            self.mav.sendHeartbeats(udp=False)
            self.mav.sendHeartbeats(udp=True)
            self.sitl_actuation = torch.zeros(self.scene.num_envs,4, 3, device="cuda",)
            self.sitl_moment = torch.zeros(self.scene.num_envs,4, 3, device="cuda",)

            self.grace_period = torch.zeros(self.scene.num_envs, dtype=torch.bool)
            self.grace_counter = torch.zeros(self.scene.num_envs, dtype=torch.int32)
            self.grace_steps = int((2.0/4.0) / self.cfg.sim.dt)
            self.grace_armed_mask = torch.zeros(self.scene.num_envs, dtype=torch.bool)

            self.arm_env_ids = (~self.grace_armed_mask).nonzero(as_tuple=True)[0].tolist()

        #mission planner
        self.rngen = torch.Generator(device="cuda").manual_seed(1)
        self.actgen = torch.Generator(device="cuda").manual_seed(10)
        self.primitive = ActionPrimitive(
            actions=[
                RandomWalk,
                RandomSphereOffset,
                HoldPosition,
            ],
            dim=3,
            num_envs=self.scene.num_envs,
            device="cuda",
            min_duration=100,
            max_duration=800,
            generator=self.actgen
        )

    def _setup_scene(self):
        self.drone = Drone(self.scene.cfg.num_envs)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=True)
        self.event_manager = EventManager(self.cfg.events, self)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        #light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        #light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        #self._actions = actions.clone()
        self.loop_counter += 1
        #env_timestamp = self.episode_length_buf.reshape(self.episode_length_buf.shape[0], 1) * self.cfg.sim.dt
        if self.loop_counter % self.steps_per_sample == 0:
            if self.sitl:
                self.grace_counter[self.grace_period] += 1
                done = self.grace_counter >= self.grace_steps
                self.grace_period[done] = False

                actuation = self.mav.recvActuation()
                actuation_t = torch.tensor(actuation, device=self.device)

                self.sitl_actuation[..., 2] = actuation_t * 12.5
                self.sitl_moment[..., 2] = self.sitl_actuation[..., 2] * 0.09
                self.sitl_moment[..., 0] *= -1.0
                self.sitl_moment[..., 2] *= -1.0

                #self.grace_armed_mask = self.mav.armed & ~self.grace_period
                #grace_armed_mask signals the script when the arming command is allowed to
                #be sent
                self.grace_armed_mask = self.grace_period
                self.arm_env_ids = (~self.grace_armed_mask).nonzero(as_tuple=True)[0].tolist()
                #if self.loop_counter % 4 == 0:
                #    print("arm ", self.mav.armed, 
                #          " grace_armed  ", self.grace_armed_mask,
                #          " grace_counter    ", self.grace_counter,
                #          " grace_steps    ", self.grace_steps
                #          )

                self.drone.setpoint = self.primitive.step(self.scene.articulations["drone"].data.root_com_pos_w, active_mask=(~self.grace_armed_mask).to(self.device))
            else:
                self.drone.setpoint = self.primitive.step(self.scene.articulations["drone"].data.root_com_pos_w)

            drone_data = self.scene.articulations["drone"].data
            imu_data = self.scene.sensors["imu"].data

            acc =  imu_data.lin_acc_b.detach().clone()
            gyro =  imu_data.ang_vel_b.detach().clone()
            pos = drone_data.root_com_pos_w.detach().clone()                      # (N, 3)
            quat = drone_data.root_com_quat_w.detach().clone()                    # (N, 4) [w, x, y, z]
            lin_vel = drone_data.root_com_lin_vel_b.detach().clone()              # (N, 3)
            ang_vel = drone_data.root_com_ang_vel_b.detach().clone()              # (N, 3)
            setpoint = self.drone.setpoint.detach().clone()

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

            #sending mavlink stuffs
            if self.sitl:
                timestamp = int(self.sim.current_time * 1e6) & 0xFFFFFFFF
                self.mav.sendImu(
                        timestamp,
                        acc,
                        gyro,
                        )
                #TODO : send odometry per environment, and potentially only during
                #the grace period
                #TODO : also collect data on pure EKF2
                self.mav.sendOdometry(
                        timestamp,
                        pos,
                        quat_frd,
                        lin_vel,
                        ang_vel,
                        )
                self.mav.sendPositionTargets(
                        timestamp,
                        setpoint, 
                        udp=True
                        )
                #self.arm_env_ids signals which env can be armed
                self.mav.arm(force=False, udp=True, env_ids=self.arm_env_ids)
                #disarm_env_ids signals which env should be kept disarmed
                disarm_env_ids = self.grace_armed_mask.nonzero(as_tuple=True)[0].tolist()
                self.mav.disarm(force=True, udp=True, env_ids=disarm_env_ids)
                _temp_odom = torch.as_tensor(
                        self.mav.recvOdometry(udp=True),
                        device=self.device,
                        dtype=torch.float32,)
                print(_temp_odom[..., 2])
            #use simulation state and controller for not sitl
            else:
                obs = torch.cat([
                    pos, #more testing for independent pos/ remove it from training input
                    lin_vel,
                    quat_frd,
                    ang_vel,
                    acc,
                    gyro,
                    self.drone.controller.thrust,
                    setpoint,
                ], dim=1)  # (N, dim)
                self.drone.controller.step(desired_pos=setpoint, states=obs[:, :13], dt=1.0/self.sampling_freq, physics_dt=self.cfg.sim.dt)

            #print("step_idx     ", self.step_idx, "  seq_len    ", self.seq_len)
            valid = self.step_idx < self.seq_len  # (N,)
            idx = self.step_idx[valid]
            if self.sitl:
                obs = torch.cat([
                    _temp_odom[..., :3]/3.0,
                    _temp_odom[..., 3:6]/0.8,
                    _temp_odom[..., 6:10],
                    _temp_odom[..., 10:13]/0.5,
                    acc/25.0,
                    actuation_t,
                    setpoint/3.0,
                ], dim=1)  # (N, dim)
            else:
                obs = torch.cat([
                    pos/3.0,
                    lin_vel/0.8,
                    quat_frd,
                    ang_vel/0.5,
                    acc/25.0,
                    self.drone.controller.thrust/9.81,
                    setpoint/3.0,
                ], dim=1)  # (N, dim)
            self.data[valid, idx] = obs[valid]
            self.step_idx[valid] += 1


    def _apply_action(self):
        #noise/disturbance, uses a scaler here
        self.drone.thrust_noise = 2e-8 * (torch.randn_like(self.drone.controller.ve_thrust, generator=self.rngen) + self.drone.controller.ve_thrust.detach().clone())
        self.drone.moment_noise = 2e-8 * (torch.randn_like(self.drone.controller.ve_moment, generator=self.rngen) * 0.09 + self.drone.controller.ve_moment.detach().clone())

        if self.sitl:
            self.scene.articulations["drone"].set_external_force_and_torque(
                self.sitl_actuation[self.arm_env_ids],
                self.sitl_moment[self.arm_env_ids],
                body_ids=self.drone.rotor_ids[0],
                is_global=False,
            )
        else:
            self.scene.articulations["drone"].set_external_force_and_torque(
                self.drone.controller.ve_thrust + self.drone.thrust_noise, 
                self.drone.controller.ve_moment + self.drone.moment_noise,
                body_ids=self.drone.rotor_ids[0], 
                is_global=False
                )

    #not used
    def _get_observations(self) -> dict:
        observations = {}
        return observations
    #not used
    def _get_rewards(self) -> torch.Tensor:
        distance_to_goal = torch.linalg.norm(self.scene.articulations["drone"].data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
                "distance_to_goal": distance_to_goal_mapped * self.step_dt,
                }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        return reward
    #not used
    def _get_dones(self):
        drone = self.scene.articulations["drone"]
        angle = torch.acos((-drone.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        died = angle > torch.deg2rad(torch.tensor(80.0, device=angle.device))
        time_out = self.episode_length_buf >= self.max_episode_length
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.scene.articulations["drone"]._ALL_INDICES

        #reset px4 instances
        if self.sitl:
            env_list = env_ids.cpu().tolist()
            self.mav.resetVehicle(
                env_ids=env_list,
                reboot=True,
                force=True,
                udp=True,
            )
            self.sitl_actuation = self.sitl_actuation * 0.0
            self.sitl_moment = self.sitl_moment * 0.0

            self.grace_period[env_ids] = True
            self.grace_counter[env_ids] = 0

        #dron physical properties randomisation
        self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=self.common_step_counter)

        #writing default states
        joint_pos = self.scene.articulations["drone"].data.default_joint_pos[env_ids].clone()
        joint_vel = self.scene.articulations["drone"].data.default_joint_vel[env_ids].clone()
        root_state = self.scene.articulations["drone"].data.default_root_state[env_ids].clone()
        root_state[:, 2] = torch.clamp(root_state[:, 2], min=0.0)
        self.scene.articulations["drone"].write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.scene.articulations["drone"].write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.scene.articulations["drone"].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.episode_length_buf[env_ids] = 0
        base_pos = self._terrain.env_origins[env_ids]

        #reset ActionPrimitive
        self.primitive.reset(env_ids, self.scene.articulations["drone"].data.root_com_pos_w,)

        #log data
        for i in env_ids:
            if self.step_idx[i] < self.seq_len:
                continue  
            t = self.data[i].detach().to("cpu", non_blocking=True).numpy()
            self.writer.write_episode(t)
            self.data[i].zero_()
            self.step_idx[i] = 0

        #just to measuer time per batch
        self.t0=self.t1
        self.t1=time.perf_counter()
        print("time : ", self.t1 - self.t0)
        print("progress : ", self.writer.batch_idx, " out of ", self.writer.num_batch)

        #condition for ending the data collection
        if self.writer.batch_idx >= self.writer.num_batch:
            print("Dataset complete. Stopping simulation...")
            self.writer.store.close()
            raise SystemExit
