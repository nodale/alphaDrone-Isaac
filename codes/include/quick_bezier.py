from dataclasses import dataclass

from tra_spline import CubicSpline
from tra_planner import LinearLocalPlanner
from quick_state import QuickState

import numpy as np

import sys, select, signal, random
import time

@dataclass
class QuickBezier(QuickState):
    pVel : float = 0.1 #projected velocity
    step : float = 1.0

    def __init__(self, address, baudrate, hostAddress='localhost', hostPort='12345', **kwargs):
        super().__init__(address=address, baudrate=baudrate, hostAddress=hostAddress, hostPort=hostPort, **kwargs)

        self.velCommand = np.zeros(2)

        #initialising MAVLink
        self.sendHeartbeat()
        self.setFlightmode('OFFBOARD')
        #self.refeed()

        self.splineList = []
        self.llp = LinearLocalPlanner(self.splineList, self.pVel)
        
        signal.signal(signal.SIGINT, self.printBezier)

        print("BEZIER ENGAGED")

    def initNudge(self):
        _initX, _initY = 0, 0

        _point_array = np.array([[_initX, _initY], [0.25 + _initX, _initY], [-0.25 + _initX, _initY], [_initX, _initY]], dtype=np.float64)
        _bez_curve = CubicSpline(p0=_point_array[0][:], p1=_point_array[1][:], p2=_point_array[2][:], p3=_point_array[3][:])
        
        self.splineList.append(_bez_curve)

    def genRandomCurve(self):
        #temp variables 
        _min_val = -1.0
        _max_val = 1.0

        _val1 = max(min(random.uniform(_min_val, _max_val), _max_val), _min_val)
        _val2 = max(min(random.uniform(_min_val, _max_val), _max_val), _min_val)
        _val3 = max(min(random.uniform(_min_val, _max_val), _max_val), _min_val)

        _py = [self.pos[1],
              _val1,
              _val2,
              _val3]

        _point_array = np.array([[self.pos[0], _py[0]], [(self.step * 0.25) + self.pos[0], _py[1]], [(self.step * 0.5) + self.pos[0], _py[2]], [(self.step * 0.75) + self.pos[0], _py[3]]], dtype=np.float64)
        _bez_curve = CubicSpline(p0=_point_array[0][:], p1=_point_array[1][:], p2=_point_array[2][:], p3=_point_array[3][:])
        self.splineList.append(_bez_curve)
      
    def getVelSetpoint(self):
        self.llp.update_position(np.array([self.pos[0], self.pos[1]]))
        self.velCommand = self.llp.get_control_target(np.array([self.pos[0], self.pos[1]]))

        self.velCommand[0] = np.clip(self.velCommand[0],-1,1) 
        self.velCommand[1] = np.clip(self.velCommand[1],-1,1)

    def _updateRefeedState(self):
        self.refeed()
        self.updateRefeedState()
        self.allocatePrev()


    def takeoff(self, z):
        print("attempting to take off")

        for i in range(100):
            _time = int(time.time() * 1e6) & 0xFFFFFFFF
            self._updateRefeedState()
            self.sendPositionTarget(_time, self.pos[0], self.pos[1], z)
            time.sleep(1/self.freq)
        self.arm()
        for i in range(200):
            _time = int(time.time() * 1e6) & 0xFFFFFFFF
            self._updateRefeedState()
            self.sendPositionTarget(_time, self.pos[0], self.pos[1], z)
            time.sleep(1/self.freq)

    def go2FirstCurve(self):
        print("press enter to engage bezier")
        while True:
            self._updateRefeedState()
            _time = int(time.time() * 1e6) & 0xFFFFFFFF
            self.sendPositionTarget(_time, self.splineList[0].p0[0], self.splineList[0].p0[1], -1.0)
            self.show({'local position': self.pos})
            time.sleep(1/self.freq)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:   
                _line = sys.stdin.readline()
                if _line.strip() == "":
                    print("BEZIER ENGAGED")
                    break

    def traverseCurve(self):
        self._updateRefeedState()
        self.getVelSetpoint()
        _currentTime = int(time.time() * 1e6) & 0xFFFFFFFF
        self.sendVelocityTarget(_currentTime, self.velCommand[0], self.velCommand[1], 0.0)
        time.sleep(1/self.freq)
        self.show({"local position": self.pos,
                  "closest point" : self.llp.closest_u
                   })

    #for debugging or visualisation
    def resetLogFiles(self):
        self.filename = 'log/pos.txt'
        with open(self.filename, 'w') as _f:
            #_f.write('x1\ty1\tz1\tx2\ty2\tz2\tx_sp\ty_sp\tz_sp\tt\n')
            _f.write('\n')

        self.filename_q = 'log/q.txt'
        with open(self.filename_q, 'w') as _fq:
            _fq.write('x1\ty1\tz1\tx2\ty2\tz2\tt\n')

        self.filename_ang = 'log/ang.txt'
        with open(self.filename_ang, 'w') as _fa:
            _fa.write('x1\ty1\tz1\tx2\ty2\tz2\tt\n')

        self.filename_v = 'log/vel.txt'
        with open(self.filename_v, 'w') as _fv:
            _fv.write('x1\ty1\tz1\tx2\ty2\tz2\tx_sp\ty_sp\tz_sp\tt\n')

        self.filename_b = 'log/b.txt'
        with open(self.filename_b, 'w') as _fb:
            _fb.write('\n')

        self.filename_bezier = 'log/bezier.txt'
        with open(self.filename_bezier, 'w') as _fbezier:
            _fbezier.write('\n')

    def printBezier(self, signum=None, frame=None):
        with open('log/bezier.txt', "w") as f:
            for cubicList in self.splineList:
                x, y = cubicList.p0
                f.write(f"{x} {y}\n")

                x, y = cubicList.p1
                f.write(f"{x} {y}\n")

                x, y = cubicList.p2
                f.write(f"{x} {y}\n")

                x, y = cubicList.p3
                f.write(f"{x} {y}\n")
                f.write("\n") 
        sys.exit(0)


    def printData(self, x1, y1, z1, x2, y2, z2, t, name):
        row = np.array([[x1, y1, z1, x2, y2, z2, t]])
        with open(name, 'a') as f:
            np.savetxt(f, row, fmt='%.6f', delimiter='\t')

    def printDataSP(self, x1, y1, z1, x2, y2, z2, x_sp, y_sp, z_sp, t, name):
        row = np.array([[x1, y1, z1, x2, y2, z2, x_sp, y_sp, z_sp, t]])
        with open(name, 'a') as f:
            np.savetxt(f, row, fmt='%.6f', delimiter='\t')

    def printAll(self):
    #quaternion debug
        self.printData(self.q[1], self.q[2], self.q[3], self.qFil[0], self.qFil[1], self.qFil[2], time.time() - self.timeBoot, self.filename_q)
    #angle debug
        self.printData(self.rot[0], self.rot[1], self.rot[2], self.rot[0], self.rot[1], self.rot[2], time.time() - self.timeBoot, self.filename_ang)
    #pos debug
        self.printDataSP(self.pos[0], self.pos[1], self.pos[2], self.pos[0], self.pos[1], self.pos[2], 0, 0, 0, time.time() - self.timeBoot, self.filename)
    #vel debug
        self.printDataSP(self.vel[0], self.vel[1], self.vel[2], self.vel[0], self.vel[1], self.vel[2], self.velCommand[0], self.velCommand[1], 0, time.time() - self.timeBoot, self.filename_v)

    def show(self, values, precision=3):
        sys.stdout.write("\033[F" * len(values))  

        for name, obj in values.items():
            if isinstance(obj, (int, float)):
                formatted = f"{obj:.{precision}f}"
            else:
                formatted = str(obj)
            print(f"DEBUG : {name} = {formatted}")
        sys.stdout.flush()
