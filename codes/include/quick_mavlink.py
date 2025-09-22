from dataclasses import dataclass

from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2

import time

@dataclass
class QuickMav:
    freq : float = 50
    timeBoot : float = 0.0

    def __init__(self, address, baudrate, **kwargs):
        self.timeBoot = time.time()
        try:
            self.master = mavutil.mavlink_connection(address, baudrate)
        except:
            print("error in __init__, MAVlink refuses to connect, maybe wrong address or baudrate")
        super().__init__(**kwargs)


    def setFreq(self, nfreq):
        self.freq = nfreq

    def sendHeartbeat(self):
        try:
            print("sending heartbeat")
            self.master.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_QUADROTOR,      # or MAV_TYPE_GENERIC
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,   # still fine
                    0,                                       # base_mode
                    0,                                       # custom_mode
                    mavutil.mavlink.MAV_STATE_ACTIVE         # system_status
                    )
            self.master.wait_heartbeat(timeout=1)
        except:
            print("sending heartbeat failed :(")

        self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,  
                0,                                             
                93,                             
                10000,                                   
                0, 0, 0, 0, 0                                  
                )

        print("MAVLINK ENGAGED")


    def setFlightmode(self, mode):
        self.master.set_mode(mode)

        print("flight mode is set to ", mode)

    def arm(self):
        self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0
                )
        print("DRONE ARMED")

    def disarm(self):
        self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0
                )
        print("DRONE DISARMED")

    def get(self, TYPE, block=True):
        return self.master.recv_match(type=TYPE, blocking=False)

    def sendOdometry(self, time, pos, q, vel, rotRates, cov1=[0.002]*21, cov2=[0.002]*21):
        vodom = mavlink2.MAVLink_odometry_message(
                time,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                mavutil.mavlink.MAV_FRAME_BODY_FRD,
                pos[0], pos[1], pos[2],
                [q[0], q[1], q[2], q[3]],
                vel[0], vel[1], vel[2],
                rotRates[0], rotRates[1], rotRates[2],
                cov1, 
                cov2,
                0,
                0,
                0
                )
        self.master.mav.send(vodom)

    def refeed(self):
        _translation = self.get('LOCAL_POSITION_NED', True)
        _ang = self.get('ATTITUDE', True)
        _q = self.get('ATTITUDE_QUATERNION', True)

        _time = int(time.time() * 1e6) & 0xFFFFFFFF

        self.sendOdometry(
                _time, 
                [_translation.x, _translation.y, _translation.z],
                [_q.q1, _q.q2, _q.q3, _q.q4],
                [_translation.vx, _translation.vy, _translation.vz],
                [_ang.rollspeed, _ang.pitchspeed, _ang.yawspeed]
                )

    def sendVelocityTarget(self, time, vx, vy, vz): 
        self.master.mav.set_position_target_local_ned_send(
                time,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,
                0, 0, 0,  #position
                vx, vy, vz,  #velocity
                0, 0, 0,  #acceleration
                0, 0  #yaw yaw_rate
                )

    def sendPositionTarget(self, time, x, y, z): 
        self.master.mav.set_position_target_local_ned_send(
                time,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,
                x, y, z,  #position
                0, 0, 0,  #velocity
                0, 0, 0,  #acceleration
                0, 0  #yaw yaw_rate
                )
