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

        n = self.N - 1
        i = torch.arange(self.N, device=self.device, dtype=dtype)
        self.C = torch.exp(
            torch.lgamma(torch.tensor(n + 1.0, device=self.device, dtype=dtype))
            - torch.lgamma(i + 1)
            - torch.lgamma(torch.tensor(n, device=self.device, dtype=dtype) - i + 1)
        )

    @staticmethod
    def random(M=1, N=3, offset=None, low=-3.00, high=3.00, device=None, dtype=torch.float32, seed=0):
        generator = torch.Generator(device=device if device is not None else "cpu")
        generator.manual_seed(seed)

        low_t = torch.tensor([0.0, low, 0.3], device=device, dtype=dtype)
        high_t = torch.tensor([high, high, 2.0], device=device, dtype=dtype)

        ctrl = (high_t - low_t) * torch.rand((M, N, 3), generator=generator, device=device, dtype=dtype) + low_t
        idx = torch.argsort(ctrl[:, :, 0], dim=1)
        ctrl = torch.gather(ctrl, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        ctrl[:, 0, :2] = torch.tensor([0.0, 0.0], device=device, dtype=dtype)

        angle = 2 * torch.pi * torch.rand((M,), generator=generator, device=device, dtype=dtype)
        c = torch.cos(angle)
        s = torch.sin(angle)
        x = ctrl[:, :, 0].clone()
        y = ctrl[:, :, 1].clone()
        ctrl[:, :, 0] = c[:, None] * x - s[:, None] * y
        ctrl[:, :, 1] = s[:, None] * x + c[:, None] * y

        if offset is not None:
            ctrl = ctrl + offset.unsqueeze(1)

        return Planner(ctrl, device=device, dtype=dtype)

    def _basis(self, t):
        t = t[:, None]
        i = torch.arange(self.N, device=self.device)[None, :]
        return self.C * (t ** i) * ((1 - t) ** (self.N - 1 - i))

    def _dbasis(self, t):
        B = self._basis(t)
        dB = torch.zeros_like(B)
        for i in range(self.N - 1):
            dB[:, i] += (self.N - 1) * (B[:, i + 1] - B[:, i])
        dB[:, -1] = -(self.N - 1) * B[:, -1]
        return dB

    def point(self, t):
        B = self._basis(t)
        return torch.einsum('mnk,mn->mk', self.ctrl, B)

    def vel(self, t):
        dB = self._dbasis(t)
        return torch.einsum('mnk,mn->mk', self.ctrl, dB)

    def update_t(self, pos, dt=0.02):
        pos = torch.as_tensor(pos, dtype=self.ctrl.dtype, device=self.device)
        p = self.point(self.t)
        v = self.vel(self.t)
        err = pos - p
        vt = torch.sum(err * v, dim=1)
        denom = torch.sum(v * v, dim=1) + 1e-8
        self.t = self.t + dt * (vt / denom)
        self.t = torch.clamp(self.t, 0.0, 1.0)
        return self.t

    def step(self, pos, dt=0.02, lookahead=0.08):
        pos = torch.as_tensor(pos, dtype=self.ctrl.dtype, device=self.device)
        self.update_t(pos, dt)
        t = torch.clamp(self.t + lookahead, 0.0, 1.0)
        return self.point(t)
