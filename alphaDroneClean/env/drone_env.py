# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import time
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.managers import EventManager, EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from env.drone_asset import Drone
from planner.primitives import (
    ActionPrimitive, HoldPosition, RandomWalk, RandomSphereOffset,
    CircularOrbit, HelixClimb, LissajousPath, ZigZag, SinusoidalWalk,
)
from data.writer import DataWriter
from sitl.mavlink import QuickMavMulti

_SEED_MULT = 2
# Vibration model: quadratic sigma(T) = d·T² + e·T + f, clamped to floor [g]
# Parameters fitted from thrust_log.csv via thesis_tools/vibration_model.py
_VIB_T_MAX = 13.0  # N — model extrapolation limit
_VIB_SIGMA_X = (-0.01168,  0.11088, -0.02476, 0.01638)  # (d, e, f, floor)
_VIB_SIGMA_Y = (-0.00363,  0.03557,  0.03776, 0.02099)
_VIB_SIGMA_Z = ( 0.00052,  0.00402,  0.04676, 0.01606)

def _vib_sigma(T: torch.Tensor, d: float, e: float, f: float, floor: float) -> torch.Tensor:
    return torch.clamp(d * T**2 + e * T + f, min=floor)


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
            "asset_cfg": SceneEntityCfg("drone", body_names="base_link"),
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
            "asset_cfg": SceneEntityCfg("drone", body_names="base_link"),
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
    episode_length_s = 25.0
    decimation = 1
    action_space = 4
    observation_space = 6
    state_space = 0
    debug_vis = False
    dt = 1.0 / 800.0
    seed = 100 * _SEED_MULT

    ui_window_class_type = DroneEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=10,
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
        super().__init__(cfg, render_mode, **kwargs)

        self.n_dim = 23
        self.max_iter = 1
        self.sampling_freq = 200.0
        self.steps_per_sample = int(round(1.0 / (self.cfg.sim.dt * self.sampling_freq)))
        self.seq_len = int((self.cfg.episode_length_s / self.steps_per_sample) / self.cfg.sim.dt)
        self.writer = DataWriter(
            path='/media/egghead/Scratch/joey/simulation_data/patient_one_data.zarr/',
            num_batch=self.max_iter,
            seq_len=self.seq_len,
            n_dim=self.n_dim,
        )
        self.t1 = 1.0
        self.t0 = 0.0

        self.data = torch.zeros(self.scene.num_envs, self.seq_len, self.n_dim, device="cuda")
        self.step_idx = torch.zeros(self.scene.num_envs, dtype=torch.long, device="cpu")
        self.drone.rotor_ids = self.scene.articulations["drone"].find_bodies("rotor_[1-4]")
        self.loop_counter = 1

        self.sitl = True
        if self.sitl:
            self.mav = QuickMavMulti(
                num_envs=self.scene.num_envs,
                tcp_base=4560,
                udp_base=14580,
            )
            self.mav.sendHeartbeats(udp=False)
            self.mav.sendHeartbeats(udp=True)
            self.sitl_actuation = torch.zeros(self.scene.num_envs, 4, 3, device="cuda")
            self.sitl_moment = torch.zeros(self.scene.num_envs, 4, 3, device="cuda")

            self.grace_period = torch.zeros(self.scene.num_envs, dtype=torch.bool)
            self.grace_counter = torch.zeros(self.scene.num_envs, dtype=torch.int32)
            self.grace_steps = int((2.0 / 4.0) / self.cfg.sim.dt)
            self.grace_armed_mask = torch.zeros(self.scene.num_envs, dtype=torch.bool)

            self.arm_env_ids = (~self.grace_armed_mask).nonzero(as_tuple=True)[0].tolist()
            self.disarm_env_ids = self.grace_armed_mask.nonzero(as_tuple=True)[0].tolist()

            self.data_ekf = torch.zeros(self.scene.num_envs, self.seq_len, self.n_dim, device="cuda")
            self.writer_ekf = DataWriter(
                path='/media/egghead/Scratch/joey/simulation_data/patient_three_data.zarr/',
                num_batch=self.max_iter,
                seq_len=self.seq_len,
                n_dim=self.n_dim,
            )

        self.rngen = torch.Generator(device="cuda").manual_seed(1 * _SEED_MULT)
        self.vib_rngen = torch.Generator(device="cuda").manual_seed(3 * _SEED_MULT)
        self.actgen = torch.Generator(device="cuda").manual_seed(10 * _SEED_MULT)
        self.noise_scale = torch.ones(self.scene.num_envs, 1, 1, device="cuda")

        #actions=[RandomWalk, RandomSphereOffset, HoldPosition,
        self.primitive = ActionPrimitive(
            actions=[RandomWalk, RandomSphereOffset, HoldPosition],
            dim=3,
            num_envs=self.scene.num_envs,
            device="cuda",
            min_duration=400,
            max_duration=800,
            generator=self.actgen,
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

    def _pre_physics_step(self, actions: torch.Tensor):
        self.loop_counter += 1

        if self.loop_counter % self.steps_per_sample == 0:
            if self.sitl:
                self.grace_counter[self.grace_period] += 1
                done = self.grace_counter >= self.grace_steps
                self.grace_period[done] = False

                actuation = self.mav.recvActuation()
                actuation_t = torch.tensor(actuation, device=self.device)
                # Invert the motor-command mapping applied by mc_johnny_control's
                # actuate_motors(): signal = 0.8238*T_kgf^0.5788 + 0.0360, i.e. the
                # command is NOT linear in thrust. Applying the forward curve here
                # reproduces the thrust the controller intended (in Newtons).
                #thrust_kgf = ((actuation_t - 0.03604325541483971) / 0.823789589308134).clamp(min=0.0) ** (1.0 / 0.578815510492838)
                #self.sitl_actuation[..., 2] = thrust_kgf * 9.81
                self.sitl_actuation[..., 2] = actuation_t * 13.0
                self.sitl_moment[..., 2] = self.sitl_actuation[..., 2] * 0.09
                # rotors 1 and 3 counter-rotate (same convention as Kxontroller)
                self.sitl_moment[:, 0, 2] *= -1.0
                self.sitl_moment[:, 2, 2] *= -1.0

                self.grace_armed_mask = self.grace_period
                self.arm_env_ids = (~self.grace_armed_mask).nonzero(as_tuple=True)[0].tolist()
                self.disarm_env_ids = self.grace_armed_mask.nonzero(as_tuple=True)[0].tolist()

                self.drone.setpoint = self.primitive.step(
                    self.scene.articulations["drone"].data.root_com_pos_w,
                    active_mask=(~self.grace_armed_mask).to(self.device),
                )
            else:
                self.drone.setpoint = self.primitive.step(
                    self.scene.articulations["drone"].data.root_com_pos_w
                )

            drone_data = self.scene.articulations["drone"].data
            imu_data = self.scene.sensors["imu"].data

            acc = imu_data.lin_acc_b.detach().clone()
            gyro = imu_data.ang_vel_b.detach().clone()
            # Use link-frame pose/velocity throughout, NOT root_com_*: the COM pose
            # carries PhysX's principal-axes-of-inertia rotation (34.7 deg yaw here,
            # from the URDF's ixy term) and the COM point sits ~6 cm below the IMU.
            # Keeping position, velocity, and attitude all in the link frame makes
            # the odometry internally consistent and referenced to the same point
            # the EKF fuses against (removes the rotation-dependent lever-arm error).
            pos = drone_data.root_link_pos_w.detach().clone()
            quat = drone_data.root_link_quat_w.detach().clone()
            lin_vel = drone_data.root_link_lin_vel_b.detach().clone()
            ang_vel = drone_data.root_link_ang_vel_b.detach().clone()
            setpoint = self.drone.setpoint.detach().clone()

            w, x, y, z = quat.unbind(dim=1)
            quat_frd = torch.stack([w, x, -y, -z], dim=1)

            # Isaac NWU → PX4 NED/FRD: negate y and z components
            pos[:, 1:] *= -1.0
            lin_vel[:, 1:] *= -1.0
            ang_vel[:, 1:] *= -1.0
            acc[:, 1:] *= -1.0
            gyro[:, 1:] *= -1.0
            setpoint[:, 1:] *= -1.0

            if self.sitl:
                timestamp = int(self.sim.current_time * 1e6) & 0xFFFFFFFF
                self.mav.sendImu(timestamp, acc, gyro)
                self.mav.sendPositionTargets(timestamp, setpoint, udp=True)
                self.mav.arm(force=False, udp=True, env_ids=self.arm_env_ids)
                self.mav.disarm(force=True, udp=True, env_ids=self.disarm_env_ids)
                # ang_vel (clean rigid-body rate), NOT gyro (noisy IMU sample): the
                # gyro belongs in HIL_SENSOR above; the odometry rate should be clean.
                self.mav.sendOdometry(time_usec=timestamp, pos=pos, quat=quat_frd, vel=lin_vel, ang_vel=ang_vel, udp=True)
                _temp_odom = torch.as_tensor(
                    self.mav.recvOdometry(udp=True),
                    device=self.device,
                    dtype=torch.float32,
                )
                print(_temp_odom[..., 2])
            else:
                obs = torch.cat([
                    pos, lin_vel, quat_frd, ang_vel, acc,
                    self.drone.controller.thrust, setpoint,
                ], dim=1)
                self.drone.controller.step(
                    desired_pos=setpoint,
                    states=obs[:, :13],
                    dt=1.0 / self.sampling_freq,
                    physics_dt=self.cfg.sim.dt,
                )

            valid = self.step_idx < self.seq_len
            idx = self.step_idx[valid]

            if self.sitl:
                obs_ekf = torch.cat([
                    _temp_odom[..., :3] / 3.0,
                    _temp_odom[..., 3:6] / 0.8,
                    _temp_odom[..., 6:10],
                    _temp_odom[..., 10:13] / 0.5,
                    acc / 25.0,
                    actuation_t,
                    setpoint / 3.0,
                ], dim=1)
                self.data_ekf[valid, idx] = obs_ekf[valid]

            obs = torch.cat([
                pos / 3.0,
                lin_vel / 0.8,
                quat_frd,
                ang_vel / 0.5,
                acc / 25.0,
                self.drone.controller.thrust / 9.81,
                setpoint / 3.0,
            ], dim=1)
            self.data[valid, idx] = obs[valid]
            self.step_idx[valid] += 1

    def _apply_action(self):
        #self.drone.thrust_noise = self.noise_scale * (
        #    torch.randn_like(self.drone.controller.ve_thrust, generator=self.rngen)
        #    + self.drone.controller.ve_thrust.detach().clone()
        #)
        #self.drone.moment_noise = self.noise_scale * (
        #    torch.randn_like(self.drone.controller.ve_moment, generator=self.rngen) * 0.09
        #    + self.drone.controller.ve_moment.detach().clone()
        #)


        if self.sitl:
            T = self.sitl_actuation[..., 2].clamp(0.0, _VIB_T_MAX)  # (num_envs, 4)
            sig = torch.stack([
                _vib_sigma(T, *_VIB_SIGMA_X),
                _vib_sigma(T, *_VIB_SIGMA_Y),
                _vib_sigma(T, *_VIB_SIGMA_Z),
            ], dim=2)  # (num_envs, 4, 3)
            vib = torch.randn(self.scene.num_envs, 4, 3, device=self.device, generator=self.vib_rngen) * sig
            self.drone.thrust_noise = self.noise_scale * vib
            self.drone.moment_noise = self.noise_scale * vib * 0.09

            self.scene.articulations["drone"].set_external_force_and_torque(
                self.sitl_actuation[self.arm_env_ids] + self.drone.thrust_noise[self.arm_env_ids],
                self.sitl_moment[self.arm_env_ids] + self.drone.moment_noise[self.arm_env_ids],
                body_ids=self.drone.rotor_ids[0],
                is_global=False,
            )
        else:
            T = self.drone.controller.ve_thrust[..., 2].clamp(0.0, _VIB_T_MAX)  # (num_envs, 4)
            sig = torch.stack([
                _vib_sigma(T, *_VIB_SIGMA_X),
                _vib_sigma(T, *_VIB_SIGMA_Y),
                _vib_sigma(T, *_VIB_SIGMA_Z),
            ], dim=2)  # (num_envs, 4, 3)
            vib = torch.randn(self.scene.num_envs, 4, 3, device=self.device, generator=self.vib_rngen) * sig
            self.drone.thrust_noise = self.noise_scale * vib
            self.drone.moment_noise = self.noise_scale * vib * 0.09

            self.scene.articulations["drone"].set_external_force_and_torque(
                self.drone.controller.ve_thrust + self.drone.thrust_noise,
                self.drone.controller.ve_moment + self.drone.moment_noise,
                body_ids=self.drone.rotor_ids[0],
                is_global=False,
            )

    def _get_observations(self) -> dict:
        return {}

    def _get_rewards(self) -> torch.Tensor:
        distance_to_goal = torch.linalg.norm(
            self.scene.articulations["drone"].data.root_pos_w, dim=1
        )
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        reward = distance_to_goal_mapped * self.step_dt
        return reward

    def _get_dones(self):
        drone = self.scene.articulations["drone"]
        angle = torch.acos((-drone.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        died = angle > torch.deg2rad(torch.tensor(60.0, device=angle.device))
        time_out = self.episode_length_buf >= self.max_episode_length
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.scene.articulations["drone"]._ALL_INDICES

        self.noise_scale[env_ids] = torch.empty(
            len(env_ids), 1, 1, device=self.device
        ).uniform_(5e-1, 1.5, generator=self.rngen)

        if self.sitl:
            env_list = env_ids.cpu().tolist()
            self.mav.resetVehicle(env_ids=env_list, reboot=True, force=True, udp=True)
            self.sitl_actuation = self.sitl_actuation * 0.0
            self.sitl_moment = self.sitl_moment * 0.0
            self.grace_period[env_ids] = True
            self.grace_counter[env_ids] = 0

        self.event_manager.apply(
            mode="reset", env_ids=env_ids, global_env_step_count=self.common_step_counter
        )

        joint_pos = self.scene.articulations["drone"].data.default_joint_pos[env_ids].clone()
        joint_vel = self.scene.articulations["drone"].data.default_joint_vel[env_ids].clone()
        root_state = self.scene.articulations["drone"].data.default_root_state[env_ids].clone()
        root_state[:, 2] = torch.clamp(root_state[:, 2], min=0.0)
        self.scene.articulations["drone"].write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.scene.articulations["drone"].write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.scene.articulations["drone"].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.episode_length_buf[env_ids] = 0
        self.primitive.reset(env_ids, self.scene.articulations["drone"].data.root_com_pos_w)

        for i in env_ids:
            if self.step_idx[i] < self.seq_len:
                continue
            t = self.data[i].detach().to("cpu", non_blocking=True).numpy()
            self.writer.write_episode(t)
            if self.sitl:
                t_ekf = self.data_ekf[i].detach().to("cpu", non_blocking=True).numpy()
                self.writer_ekf.write_episode(t_ekf)
            self.data[i].zero_()
            self.step_idx[i] = 0

        self.t0 = self.t1
        self.t1 = time.perf_counter()
        print("time : ", self.t1 - self.t0)
        print("progress : ", self.writer.batch_idx, " out of ", self.writer.num_batch)

        if self.writer.batch_idx >= self.writer.num_batch:
            print("Dataset complete. Stopping simulation...")
            self.writer.store.close()
            raise SystemExit
