from dataclasses import dataclass
import math
import torch
import numpy as np

from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2


@dataclass
class QuickMavMulti:
    num_envs: int
    tcp_base: int = 4560
    udp_base: int = 14580
    baudrate: int = 57600

    def __post_init__(self):
        self.tcp_masters = [
            mavutil.mavlink_connection(f"tcpin:localhost:{self.tcp_base+i}", self.baudrate)
            for i in range(self.num_envs)
        ]
        self.udp_masters = [
            mavutil.mavlink_connection(f"udpout:localhost:{self.udp_base+i}", self.baudrate)
            for i in range(self.num_envs)
        ]

        self.last_actuation = np.zeros((self.num_envs, 16), dtype=np.float32)
        self._actuation_initialized = np.zeros(self.num_envs, dtype=bool)

        self.last_odometry = np.zeros((self.num_envs, 13), dtype=np.float32)
        self._odometry_initialized = np.zeros(self.num_envs, dtype=bool)

        self.armed = torch.zeros(self.num_envs, dtype=torch.bool)

    def _master(self, idx, udp=False):
        return self.udp_masters[idx] if udp else self.tcp_masters[idx]

    def _get_indices(self, env_ids):
        if env_ids is None:
            return range(self.num_envs)
        return list(env_ids)

    def _sendCommandLong(self, command, env_ids=None, param1=0, param2=0,
                         param3=0, param4=0, param5=0, param6=0, param7=0, udp=False):
        masters = self.udp_masters if udp else self.tcp_masters
        for idx in self._get_indices(env_ids):
            master = masters[idx]
            master.mav.command_long_send(
                master.target_system, master.target_component,
                command, 0, param1, param2, param3, param4, param5, param6, param7,
            )

    def sendHeartbeats(self, udp=False):
        for master in (self.udp_masters if udp else self.tcp_masters):
            try:
                master.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0,
                    mavutil.mavlink.MAV_STATE_ACTIVE,
                )
                master.wait_heartbeat(timeout=1e-5)
            except:
                print("connection failed")

    def arm(self, env_ids=None, force=False, udp=False):
        self._sendCommandLong(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            env_ids=env_ids, param1=1, param2=21196 if force else 0, udp=udp,
        )

    def disarm(self, env_ids=None, force=False, udp=False):
        self._sendCommandLong(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            env_ids=env_ids, param1=0, param2=21196 if force else 0, udp=udp,
        )

    def sendOdometry(self, time_usec, pos, quat, vel, ang_vel, udp=False):
        masters = self.udp_masters if udp else self.tcp_masters
        for i, master in enumerate(masters):
            msg = mavlink2.MAVLink_odometry_message(
                int(time_usec),
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                mavutil.mavlink.MAV_FRAME_BODY_FRD,
                float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]),
                quat[i].tolist(),
                float(vel[i, 0]), float(vel[i, 1]), float(vel[i, 2]),
                float(ang_vel[i, 0]), float(ang_vel[i, 1]), float(ang_vel[i, 2]),
                [0.01] * 21, [0.01] * 21, 0, 0, 0,
            )
            master.mav.send(msg)

    def sendImu(self, time_usec, acc, gyro, udp=False):
        masters = self.udp_masters if udp else self.tcp_masters
        for i, master in enumerate(masters):
            master.mav.hil_sensor_send(
                int(time_usec),
                float(acc[i, 0]), float(acc[i, 1]), float(acc[i, 2]),
                float(gyro[i, 0]), float(gyro[i, 1]), float(gyro[i, 2]),
                0, 0, 0, 0, 0, 0, 0,
                0b0000000111111,
            )

    def sendFakeGPS(self, time_usec, pos, udp=False):
        masters = self.udp_masters if udp else self.tcp_masters
        lat0 = 71.0 * 1e7
        lon0 = -40.0 * 1e7
        alt0 = 500.0 * 1e3
        R = 6378137.0

        for i, master in enumerate(masters):
            x = float(pos[i, 0])
            y = float(pos[i, 1])
            z = float(pos[i, 2])
            dlat = y / R
            dlon = x / (R * math.cos(math.radians(lat0 / 1e7)))
            lat = lat0 + math.degrees(dlat) * 1e7
            lon = lon0 + math.degrees(dlon) * 1e7
            alt = alt0 + z * 1000.0
            master.mav.hil_gps_send(
                int(time_usec), 3, int(lat), int(lon), int(alt),
                0, 0, 0, 0, 0, 0, 65535, 255, 0, 36000,
            )

    def sendPositionTargets(self, time_usec, pos, udp=False):
        masters = self.udp_masters if udp else self.tcp_masters
        for i, master in enumerate(masters):
            master.mav.set_position_target_local_ned_send(
                int(time_usec),
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,
                float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]),
                0, 0, 0, 0, 0, 0, 0, 0,
            )

    def sendVelocityTargets(self, time_usec, vel, udp=False):
        masters = self.udp_masters if udp else self.tcp_masters
        for i, master in enumerate(masters):
            master.mav.set_position_target_local_ned_send(
                int(time_usec),
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,
                0, 0, 0,
                float(vel[i, 0]), float(vel[i, 1]), float(vel[i, 2]),
                0, 0, 0, 0, 0,
            )

    def recvActuation(self, udp=False):
        masters = self.udp_masters if udp else self.tcp_masters
        for i, master in enumerate(masters):
            msg = master.recv_match(type="HIL_ACTUATOR_CONTROLS", blocking=False)
            if msg is not None:
                controls = np.array(msg.controls, dtype=np.float32)
                self.last_actuation[i] = controls
                self._actuation_initialized[i] = True
                self.armed[i] = bool(int(msg.mode) & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            elif not self._actuation_initialized[i]:
                self.last_actuation[i] = np.zeros(16, dtype=np.float32)
        return self.last_actuation[..., :4].copy()

    def recvOdometry(self, udp=False):
        masters = self.udp_masters if udp else self.tcp_masters
        for i, master in enumerate(masters):
            msg = master.recv_match(type="ODOMETRY", blocking=False)
            if msg is not None:
                odom = np.array([
                    msg.x, msg.y, msg.z,
                    msg.vx, msg.vy, msg.vz,
                    *msg.q,
                    msg.rollspeed, msg.pitchspeed, msg.yawspeed,
                ], dtype=np.float32)
                self.last_odometry[i] = odom
                self._odometry_initialized[i] = True
            elif not self._odometry_initialized[i]:
                self.last_odometry[i].fill(0.0)
        return self.last_odometry.copy()

    def printEstimatorStatus(self, udp=False):
        masters = self.udp_masters if udp else self.tcp_masters
        for i, master in enumerate(masters):
            msg = master.recv_match(type="ESTIMATOR_STATUS", blocking=False)
            if msg is None:
                print(f"[{i}] No ESTIMATOR_STATUS")
            else:
                print(msg)

    def printHeartbeats(self, udp=False):
        masters = self.udp_masters if udp else self.tcp_masters
        for i, master in enumerate(masters):
            msg = master.recv_match(type="HEARTBEAT", blocking=False)
            if msg is None:
                print(f"[{i}] No HEARTBEAT")
            else:
                print(
                    f"[{i}] HEARTBEAT | "
                    f"sysid={msg.get_srcSystem()} "
                    f"compid={msg.get_srcComponent()} "
                    f"mode={msg.custom_mode} "
                    f"base_mode={msg.base_mode} "
                    f"status={msg.system_status}"
                )

    def rebootAutopilot(self, env_ids=None, force=True, udp=False):
        self._sendCommandLong(
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            env_ids=env_ids, param1=1, param6=20190226 if force else 0, udp=udp,
        )

    def resetVehicle(self, env_ids=None, reboot=True, force=True, udp=True):
        self.armed[env_ids] = False
        self.disarm(env_ids=env_ids, force=force, udp=udp)
        if reboot:
            self.rebootAutopilot(env_ids=env_ids, force=force, udp=udp)
            masters = self.udp_masters if udp else self.tcp_masters
            for idx in self._get_indices(env_ids):
                masters[idx].wait_heartbeat(timeout=1e-4)
