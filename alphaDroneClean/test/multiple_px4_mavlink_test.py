from dataclasses import dataclass
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2

import time


@dataclass
class QuickMavMulti:
    num_envs: int = 1
    tcp_base: int = 4560
    udp_base: int = 14580

    def __post_init__(self):
        self.timeBoot = time.time()

        self.tcp_masters = []
        self.udp_masters = []

        self._init_tcp()
        self._init_udp()

    def _init_tcp(self):
        for i in range(self.num_envs):
            addr = f"tcpin:localhost:{self.tcp_base + i}"
            print(f"[TCP] connecting -> {addr}")

            master = mavutil.mavlink_connection(addr)
            self.tcp_masters.append(master)

    def _init_udp(self):
        for i in range(self.num_envs):
            addr = f"udp:0.0.0.0:{self.udp_base + i}"
            print(f"[UDP] connecting -> {addr}")

            master = mavutil.mavlink_connection(addr)
            self.udp_masters.append(master)

    def sendHeartbeat(self, idx: int = 0, use_udp: bool = False):
        master = self._get_master(idx, use_udp)

        try:
            master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                mavutil.mavlink.MAV_STATE_ACTIVE
            )

            master.wait_heartbeat(timeout=1)
            print(f"[{idx}] heartbeat OK")

        except Exception as e:
            print(f"[{idx}] heartbeat failed:", e)

    def get(self, idx: int, msg_type: str, use_udp: bool = False):
        master = self._get_master(idx, use_udp)
        return master.recv_match(type=msg_type, blocking=False)

    def sendOdometry(self, idx: int, time_usec, pos, q, vel, rotRates, use_udp: bool = False):
        master = self._get_master(idx, use_udp)

        msg = mavlink2.MAVLink_odometry_message(
            time_usec,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            pos[0], pos[1], pos[2],
            [q[0], q[1], q[2], q[3]],
            vel[0], vel[1], vel[2],
            rotRates[0], rotRates[1], rotRates[2],
            [0.01] * 21,
            [0.01] * 21,
            0, 0, 0
        )

        master.mav.send(msg)

    def sendVelocityTarget(self, idx: int, time_usec, vx, vy, vz, use_udp: bool = False):
        master = self._get_master(idx, use_udp)

        master.mav.set_position_target_local_ned_send(
            time_usec,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )

    def sendPositionTarget(self, idx: int, time_usec, x, y, z, use_udp: bool = False):
        master = self._get_master(idx, use_udp)

        master.mav.set_position_target_local_ned_send(
            time_usec,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,
            x, y, z,
            0, 0, 0,
            0, 0, 0,
            0, 0
        )

    def sendSimSensors(self, idx: int, time_usec, _acc, _gyro, use_udp: bool = False):
        master = self._get_master(idx, use_udp)

        master.mav.hil_sensor_send(
                time,
                _acc[0], -_acc[1], -_acc[2],
                _gyro[0], -_gyro[1], -_gyro[2],
                0, 0, 0,
                0, 0,
                0, 0,
                0b0000000111111
                )

    def sendFakeGPS(self, idx : int, time_usec, pos, use_udp: bool = False):
        master = self._get_master(idx, use_udp)

        _lat0, _lon0, _alt0 = 71 * 1e7, -40 * 1e7, 500 * 1e3

        _r = 6378137.0 
        _dlat = pos[1] / _r
        _dlon = pos[0] / (_r * math.cos(math.radians(_lat0)))
        _lat = _lat0 + math.degrees(_dlat)
        _lon = _lon0 + math.degrees(_dlon)
        _alt = _alt0 + pos[2] 

        master.mav.hil_gps_send(
                time, 
                3,
                int(_lat), 
                int(_lon), 
                int(_alt), 
                0, 
                0, 
                0, 
                0, 
                0, 
                0, 
                65535, 
                255, 
                0, 
                36000 
                )

    def _get_master(self, idx: int, use_udp: bool):
        if use_udp:
            return self.udp_masters[idx]
        return self.tcp_masters[idx]



# only for testing
def main():
    mav = QuickMavMulti(num_envs=5)

    time.sleep(1)

    for i in range(5):
        mav.sendHeartbeat(i)

    mav.sendVelocityTarget(
        idx=0,
        time_usec=int(time.time() * 1e6) & 0xFFFFFFFF,
        vx=1.0,
        vy=0.0,
        vz=0.0
    )

    msg = mav.get(0, "LOCAL_POSITION_NED")
    print("MSG:", msg)


if __name__ == "__main__":
    main()
