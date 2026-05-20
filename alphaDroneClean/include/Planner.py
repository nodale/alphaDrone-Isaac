import torch

class Planner:
    def __init__(self, ctrl, device=None, dtype=torch.float32):
        """
        ctrl: (M, N, 3) control points per environment
        """
        self.device = device or torch.device("cpu")

        self.ctrl = torch.as_tensor(ctrl, dtype=dtype, device=self.device)
        self.M, self.N, _ = self.ctrl.shape

        self.t = torch.zeros(self.M, dtype=dtype, device=self.device)

        # binomial coefficients for Bernstein basis
        n = self.N - 1
        i = torch.arange(self.N, device=self.device, dtype=dtype)

        self.C = torch.exp(
            torch.lgamma(torch.tensor(n + 1.0, device=self.device, dtype=dtype))
            - torch.lgamma(i + 1)
            - torch.lgamma(torch.tensor(n, device=self.device, dtype=dtype) - i + 1)
        )

    @staticmethod
    def random(M, N, low=0.0, high=1.0, device=None, dtype=torch.float32):
        low = torch.tensor([low, low, 0.5], device=device, dtype=dtype)
        high = torch.tensor([high, high, 1.5], device=device, dtype=dtype)

        ctrl = (high - low) * torch.rand((M, N, 3), device=device, dtype=dtype) + low
        return Planner(ctrl, device=device, dtype=dtype)

    def _basis(self, t):
        # t: (M,)
        t = t[:, None]                      # (M,1)
        i = torch.arange(self.N, device=self.device)[None, :]  # (1,N)

        return self.C * (t ** i) * ((1 - t) ** (self.N - 1 - i))

    def _dbasis(self, t):
        B = self._basis(t)  # (M,N)
        dB = torch.zeros_like(B)

        for i in range(self.N - 1):
            dB[:, i] += (self.N - 1) * (B[:, i + 1] - B[:, i])

        dB[:, -1] = -(self.N - 1) * B[:, -1]
        return dB

    def point(self, t):
        B = self._basis(t)  # (M,N)
        return torch.einsum('mnk,mn->mk', self.ctrl, B)

    def vel(self, t):
        dB = self._dbasis(t)
        return torch.einsum('mnk,mn->mk', self.ctrl, dB)

    def update_t(self, pos, dt=0.02):
        pos = torch.as_tensor(pos, dtype=self.ctrl.dtype, device=self.device)

        p = self.point(self.t)
        v = self.vel(self.t)

        err = pos - p  # (M,3)

        vt = torch.sum(err * v, dim=1)
        denom = torch.sum(v * v, dim=1) + 1e-8

        self.t = self.t + dt * (vt / denom)
        self.t = torch.clamp(self.t, 0.0, 1.0)

        return self.t

    def step(self, pos, mode="pos_vel", dt=0.02, lookahead=0.05):
        pos = torch.as_tensor(pos, dtype=self.ctrl.dtype, device=self.device)

        self.update_t(pos, dt)

        t = torch.clamp(self.t + lookahead, 0.0, 1.0)

        #out = {}

        #if mode in ("pos", "pos_vel"):
        #    out["pos"] = self.point(t)

        #if mode in ("vel", "pos_vel"):
        #    v = self.vel(t)
        #    out["vel"] = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)

        #return out
        return self.point(t)

def main():
    bz = Planner.random(M=8, N=5, low=-5, high=5)
    out = bz.step(current_pos, mode="pos_vel")
    pos_setpoints = out.get("pos")
    vel_setpoints = out.get("vel")


if __name__ == "__main__":
    main()

#prompt:
#can you generate a python class to generate bezier curves based on N control points using a function, not pre-generated? we also want it to be vectorised for the size of M. We want to use it for navigation setpoints, Using the current positions size (M, 3) of M environments, analyse te current state in the bezier curve and return the setpoints to be followed based on the environments' bezier curves. each M has to have different 3D bezier curve. We also want the options for the setpoints to be positions only, velocities only, or position_velocity. This will be repeated at each loop, using the current state, to get a setpoint to following the bezier curve. we also want to be able to genereate new bezier curves using random number generator. make the code as short and simple as possible.
