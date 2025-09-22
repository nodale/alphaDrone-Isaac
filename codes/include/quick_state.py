from dataclasses import dataclass

from quick_mavlink import QuickMav
from quick_extCom import QuickExtCom

import numpy as np
import time

@dataclass
class QuickState(QuickMav, QuickExtCom):
    alphaPos : float = 1.0
    alphaVel : float = 1.0
    alphaQ  : float = 1.0
    alphaRot : float = 1.0

    dt : float = 0.1   

    def __init__(self, **kwargs):
        self.pos = np.zeros(3)       # x, y, z
        self.vel = np.zeros(3)       # vx, vy, vz
        self.q = np.zeros(4)         # quaternion: w, x, y, z
        self.rot = np.zeros(3)       # roll, pitch, yaw
        self.rotRates = np.zeros(3)  # roll, pitch, yaw rates

        self.posP = np.zeros(3)  # x, y, z
        self.velP = np.zeros(3)  # vx, vy, vz
        self.rotP = np.zeros(3)  # roll, pitch, yaw
        self.qP = np.zeros(4)    # quaternion: w, x, y, z

        self.posFil = np.zeros(3)       # x, y, z
        self.velFil = np.zeros(3)       # vx, vy, vz
        self.qFil = np.zeros(4)         # quaternion: w, x, y, z
        self.rotFil = np.zeros(3)       # roll, pitch, yaw
        self.rotRatesFil = np.zeros(3)  # roll, pitch, yaw rates
        
        self.timeC = time.time()
        self.timeP = time.time()

        super().__init__(**kwargs)

    def updateRefeedState(self):
        _translation = self.get('LOCAL_POSITION_NED', True)
        _ang = self.get('ATTITUDE', True)
        _q = self.get('ATTITUDE_QUATERNION', True)

        self.pos = [_translation.x, _translation.y, _translation.z]
        self.q = [_q.q1, _q.q2, _q.q3, _q.q4]
        self.vel = [_translation.vx, _translation.vy, _translation.vz]
        self.rot = [_ang.roll, _ang.pitch, _ang.yaw]
        self.rotRates = [_ang.rollspeed, _ang.pitchspeed, _ang.yawspeed]

        self.timeC = time.time()
        self.dt = self.timeC - self.timeP
        self.timeP = self.timeC

    def updateViconState(self):
        try:
            self.unpacker = msgpack.Unpacker(raw=False)

            _chunk = self.conn.recv(2048)
            
            self.unpacker.feed(_chunk)

            for _msg in self.unpacker:
                if isinstance(msg, list) and len(_msg) > 0 and isinstance(_msg[0], dict):
                    _data = _msg[0]
                    _translation, _translation_flag = _data['translation']
                    _quaternion, _quaternion_flag = _data['quanternion']
                    _velocity = _data['velocity']

                    #unpack data here
                    self.pos = _translation
                    self.vel = _velocity
                    self.q = _quaternion
                    self.q2Euler()
                    self.getEulerRates()

            self.timeC = time.time()
            self.dt = self.timeC - self.timeP
            self.timeP = self.timeC
        except:
            print("failed unpacking vicon data")


    def getEulerRates(self):
        _q1 = np.array([self.q[0], self.q[1], self.q[2], self.q[3]])
        _q2 = np.array([self.qP[0], self.qP[1], self.qP[2], self.qP[3]])
        
        _q_conj = np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]])
        
        _w1, _x1, _y1, _z1 = _q_conj
        _w2, _x2, _y2, _z2 = _q2
        _q_rel = np.array([
            _w1*_w2 - _x1*_x2 - _y1*_y2 - _z1*_z2,
            _w1*_x2 + _x1*_w2 + _y1*_z2 - _z1*_y2,
            _w1*_y2 - _x1*_z2 + _y1*_w2 + _z1*_x2,
            _w1*_z2 + _x1*_y2 - _y1*_x2 + _z1*_w2
        ])
        _q_rel /= np.linalg.norm(_q_rel)
        
        _angle = 2*np.arccos(np.clip(_q_rel[0], -1, 1))
        if _angle < 1e-8:
            return np.zeros(3)
        _axis = _q_rel[1:] / np.sin(angle/2)
        _omega = (_angle/self.dt) * _axis
        
        _w, _x, _y, _z = _q1
        _roll  = np.arctan2(2*(_w*_x + _y*_z), 1 - 2*(_x*_x + _y*_y))
        _pitch = np.arcsin(np.clip(2*(_w*_y - _z*_x), -1, 1))
        
        _phi, _theta = _roll, _pitch
        _T = np.array([
            [1, np.sin(_phi)*np.tan(_theta), np.cos(_phi)*np.tan(_theta)],
            [0, np.cos(_phi),              -np.sin(_phi)],
            [0, np.sin(_phi)/np.cos(_theta), np.cos(_phi)/np.cos(_theta)]
        ])

        self.rotRates = _T @ _omega 

    def q2Euler(self):
        _sinr_cosp = 2 * (self.q[0] * self.q[1] + self.q[2] * self.q[3])
        _cosr_cosp = 1 - 2 * (self.q[1]**2 + self.q[2]**2)
        self.rot[0] = math.atan2(_sinr_cosp, _cosr_cosp)

        _sinp = 2 * (self.q[0] * self.q[2] - self.q[3] * self.q[1])
        _sinp = max(-1.0, min(1.0, _sinp))  #clamp
        self.rot[1] = math.asin(_sinp)

        _siny_cosp = 2 * (self.q[0] * self.q[3] + self.q[1] * self.q[2])
        _cosy_cosp = 1 - 2 * (self.q[2]**2 + self.q[3]**2)
        self.rot[2] = math.atan2(_siny_cosp, _cosy_cosp)

    def allocatePrev(self):
        self.posP = self.pos
        self.velP = self.vel
        self.qP = self.q
        self.rotP = self.rot

    def lpf(new_value, prev_value, alpha=0.6):
        return alpha * new_value + (1 - alpha) * prev_value

    def filter(self, position=False, velocity=False, quat=False, rotation=False):
        if position==True:
            for i in self.pos:
                self.posFil[i] = self.lpf(self.pos[i], self.posP[i], self.alphaPos)

        if velocity==True:
            for i in self.vel:
                self.velFil[i] = self.lpf(self.vel[i], self.velP[i], self.alphaVel)

        if quat==True:
            for i in self.quat:
                self.qFil[i] = self.lpf(self.q[i], self.qP[i], self.alphaQ)

        if rotation==True:
            for i in self.rot:
                self.rotFil[i] = self.lpf(self.rot[i], self.rotP[i], self.alphaRot)

    #TODO
    #create a filter based on successive gradient
