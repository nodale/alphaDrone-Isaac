import yaml
import torch
from pathlib import Path

_YAML_PATH = Path(__file__).parent / "controller.yaml"


@torch.jit.script
def quat_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


class Kilter:

    def __init__(self, num_envs, device="cuda"):

        with open(_YAML_PATH, "r") as f:
            params = yaml.safe_load(f)

        self.device = device
        self.num_envs = num_envs

        self.A_d = torch.tensor(params["A_d"], dtype=torch.float32, device=device)
        self.B_d = torch.tensor(params["B_d"], dtype=torch.float32, device=device)
        self.H = torch.eye(12, dtype=torch.float32, device=device)
        self.Q = torch.eye(12, device=device, dtype=torch.float32) * 1e-3
        self.R = torch.eye(12, device=device, dtype=torch.float32) * 1e-3
        self.P = torch.eye(12, device=device, dtype=torch.float32).expand(num_envs, 12, 12).clone()
        self.state_p = torch.zeros((num_envs, 12), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, state, u):
        state_e = (
            torch.matmul(self.state_p, self.A_d.T)
            + torch.matmul(u, self.B_d.T)
        )

        AP = torch.matmul(self.A_d, self.P)
        self.P = torch.matmul(AP, self.A_d.T) + self.Q

        S = torch.matmul(self.H, torch.matmul(self.P, self.H.T)) + self.R
        K = torch.linalg.solve(
            S,
            torch.matmul(self.H, self.P).transpose(-1, -2)
        ).transpose(-1, -2)

        innovation = state - torch.matmul(state_e, self.H.T)
        self.state_p = state_e + torch.matmul(K, innovation.unsqueeze(-1)).squeeze(-1)

        I = torch.eye(12, device=self.device, dtype=torch.float32)
        self.P = torch.matmul(I - torch.matmul(K, self.H), self.P)


class Kxontroller:

    def __init__(self, num_envs, device="cuda", freq=200, esc_delay=1.0 / 400.0):

        with open(_YAML_PATH, "r") as f:
            params = yaml.safe_load(f)

        self.device = device
        self.num_envs = num_envs
        self.control_dt = 1.0 / freq
        self.esc_delay = esc_delay

        self.K = torch.tensor(params["K_LQR"], dtype=torch.float32, device=device)
        self.P_LQR = torch.tensor(params["P_LQR"], dtype=torch.float32, device=device)

        self.u_eq = torch.tensor(
            [9.37, 9.47, 8.89, 8.79],
            dtype=torch.float32,
            device=device,
        )

        self.thrust = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        self.ve_thrust = torch.zeros((num_envs, 4, 3), dtype=torch.float32, device=device)
        self.ve_moment = torch.zeros((num_envs, 4, 3), dtype=torch.float32, device=device)

        self.kilter = Kilter(num_envs=num_envs, device=device)

    @torch.no_grad()
    def step(self, desired_pos, states, dt, physics_dt=1.0 / 800.0):
        pos = states[:, 0:3]
        vel = states[:, 3:6]
        quat = states[:, 6:10]
        ang_vel = states[:, 10:13]

        att = quat_to_euler_xyz(quat)
        state = torch.cat((pos, vel, att, ang_vel), dim=-1)
        self.kilter.update(state, self.thrust)

        state_offset = torch.cat((desired_pos - pos, -vel, -att, -ang_vel), dim=-1)

        control = self.u_eq.unsqueeze(0) - torch.matmul(state_offset, self.K.T)

        _alpha = physics_dt / self.esc_delay
        self.thrust += _alpha * (control - self.thrust.detach().clone())
        self.thrust.clamp_(min=0.0, max=13.25)

        self.ve_thrust.zero_()
        self.ve_thrust[:, :, 2] = self.thrust.detach().clone()

        self.ve_moment.zero_()
        self.ve_moment[:, :, 2] = self.thrust.detach().clone() * 0.09
        self.ve_moment[:, 0, 2] *= -1.0
        self.ve_moment[:, 2, 2] *= -1.0
