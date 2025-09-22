from dataclasses import dataclass
from typing import Any, Dict

import argparse
import sys
import carb
import time
import math
import numpy as np

from isaacsim import SimulationApp
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.prims import SinglePrim, GeometryPrim, RigidPrim

from pxr import Gf

@dataclass
class QuickIsaac(QuickBezier):
    controllableList: dict[str, Any] = field(default_factory=dict)
    rigidPrimList: dict[str, Any] = field(default_factory=dict)
    sinArtList: dict[str, Any] = field(default_factory=dict)

    def __init__(self, assetsPath, address='localhost:14550', baudrate=57600, **kwargs):
        super().__init__(address=address, baudrate=baudrate, **kwargs)

        self.simApp = SimulationApp({"headless": False})
        self.setAssetsPath(path=assetsPath)
        self.initialiseFlatWorld()

        print("simulation initialisation is done successfully\n")

    def __del__(self):
        self.simApp.close()

    def initialiseFlatWorld(self):
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

    def setAssetsPath(self, path):
        self.assetsPath = path

        if self.assetsPath is None:
            self.simApp.close()
            print("assets path is not found")
            sys.exit()
    
    def addControllable(self, name, path):
        _addRef = add_reference_to_stage(usd_path=self.assetsPath + path, primt_path=f"/World/{name}") 
        self.controllableList[name] = _addRef

        _addSinArt = SingleArticulation(prmi_path="/World/{name}", name=f"articulation_{name}")
        self.sinArtList[name] = _addSinArt
    
        _addRigPrim = RigidPrim(prim_paths_expr=f"/World/{name}/rotor_*")
        self.world.scene.add(_addRigPrim)
        self.rigidPrimList[name] = _addRigPrim

    def applyForcesAndTorques(self, name, forceVect, torqueVect, indices):
        self.rigidPrimList[name].apply_forces_and_torques_at_pos(
                indices=indices,
                forces=forceVect,
                torques=torqueVect,
                is_global=False,
                }

    def actuateDrone(self, name):
        for i in range(self.rigidPrimList[name].count):
            applyForcesAndTorques(
                    name,
                    [0, self.actOut[i] * 10, 0],
                    [0, self.actOut[i] * 10, 0],
                    i)

    def resetWorld(self):
        self.world.reset()

    def step(self):
        self.world.step(render=True)


    def initSimState(self, name):
        self.simPos, self.simQ = self.geoPrimList[name].get_local_pose()
        self.simVel = self.geoPrimList[name].get_linear_velocity()
        self.simAngVel = self.geoPrimList[name].get_angular_velocity()

        self.simPosP, self.simQP = self.geoPrimList[name].get_local_pose()
        self.simVelP = self.geoPrimList[name].get_linear_velocity()
        self.simAngVelP = self.geoPrimList[name].get_angular_velocity()

        self.timeC = time.time()
        self.timeP = time.time()
        self.dt = 0.1
        
        self.propellerJoints = [0, 1, 2, 3] 

        self.thrustVect = np.zeros([4,3])

    def getSimState(self):
        self.simPos, self.simQ = self.simPosP, self.simQP
        self.simVel = self.simVelP
        self.simAngVel = self.simAngVelP

        self.simPos, self.simQ = self.geoPrimList[name].get_local_pose()
        self.simVel = self.geoPrimList[name].get_linear_velocity()
        self.simAngVel = self.geoPrimList[name].get_angular_velocity()

        self.timeC = time.time()
        self.dt = self.timeC - self.timeP
        self.timeP = self.timeC

    #overwrites takeoff() from QuickBezier
    def takeoff(self, z):
        print("attempting to take off")

        for i in range(100):
            _time = int(time.time() * 1e6) & 0xFFFFFFFF
            self.getSimState()
            self.sendPositionTarget(_time, self.pos[0], self.pos[1], z)
            time.sleep(1/self.freq)
        self.arm()
        for i in range(200):
            _time = int(time.time() * 1e6) & 0xFFFFFFFF
            self.getSimState()
            self.sendPositionTarget(_time, self.simPos[0], self.simPos[1], z)
            time.sleep(1/self.freq)

    def addNoise(self, obj, center=0.0, amplitude=0.008, dim=3):
        obj += np.random.normal(center, amplitude, dim) 

    def getAccelerometer(self):
        _gf_quat = Gf.Quatf(simQ[3], Gf.Vec3f(simQ[0], simQ[1], simQ[2]))
        _R = gf_quat.GetRotation().GetMatrix()
        _accWorld = (np.array(self.simVel) - np.array(self.simVelP)) / self.dt
        self.simAcc = _R.T @ (_accWorld - self.accField)

        #addNoise(self.simAcc)

    def getGyroscope(self):
        _gf_quat = Gf.Quatf(simQ[3], Gf.Vec3f(simQ[0], simQ[1], simQ[2]))
        _R = gf_quat.GetRotation().GetMatrix()
        self.simGyro = _R.T @ np.array(self.simAngVel)

        #addNoise(self.simAcc)
    
    #probably not going to be used
    def getMagnetometer(self, magNED=np.array([0.2, 0.0, 0.5])):
        _gf_quat = Gf.Quatf(simQ[3], Gf.Vec3f(simQ[0], simQ[1], simQ[2]))
        _R = gf_quat.GetRotation().GetMatrix()
        self.simMag = _R.T @ magNED

        #addNoise(self.simMag)

    def getBarometer(self):
        self.simBaro = 101325 * (1 - 2.25577e-5 * self.simPos[2])**5.25588

    def sendSimSensors(self):
        self.master.mav.hil_sensor_send(
            int(time.time()*1e6),
            self.simAcc[0], self.simAcc[1], self.simAcc[2],
            self.simGyro[0], self.simGyro[1], self.simGyro[2],
            0, 0, 0,
            self.simBaro, 0,
            self.simPos[2], 28.5,
            0xFF
            )

    def sendFakeGPS(self):
        _lat0, _lon0, _alt0 = 47.397742, 8.545594, 500

        _R = 6378137.0 
        _dlat = self.simPos[1] / _R
        _dlon = self.simPos[0] / (_R * math.cos(math.radians(_lat0)))
        _lat = _lat0 + math.degrees(_dlat)
        _lon = _lon0 + math.degrees(_dlon)
        _alt = _alt0 - self.simPos[2] 

        self.master.mav.hil_gps_send(
                int(time.time() * 1e6), 
                3, 
                int(71 * 1e7), 
                int(-40 * 1e7), 
                int(500 * 1e3), 
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

    def sendFakeOdometry(self):
        _time = int(time.time() * 1e6)
        self.sendOdometry(_time, self.simPos, self.simQ, self.simVel, self.simAngVel)

    def runSimpleSensorsSim(self):
        self.getSimState()
        self.getAccelerometer()
        self.getGyroscope()
        self.getBarometer()
        self.sendFakeGPS()

        self.sendSimSensors()

    def getActuatorOutput(self):
        try:
            _actOut = self.master.recv_match(type='HIL_ACTUATOR_CONTROLS', blocking=False)
            self.actOut = np.array([_actOut.controls[0], _actOut.controls[1], _actOut.controls[2], _actOut.controls[3]])
        except:
            print("nope, no actuation")
