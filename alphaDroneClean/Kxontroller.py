import torch as torch
import numpy as np

import yaml

class Kilter:
    device = "cuda"
    Q : torch.eye = torch.eye(12, dtype=torch.float32, device=device)
    P : torch.eye = torch.eye(12, dtype=torch.float32, device=device)
    R : torch.eye = torch.eye(12, dtype=torch.float32, device=device)
    H : torch.eye = torch.eye(12, dtype=torch.float32, device=device)
    K : torch.eye = torch.eye(12, dtype=torch.float32, device=device)
    state_p : torch.tensor = torch.zeros((12,1), dtype=torch.float32, device=device)
    state_e : torch.tensor = torch.zeros((12,1), dtype=torch.float32, device=device)

    def __init__(self):
        #to init A_d and B_d and U_k
        _can_opener = open("controller.yaml", "r")
        _controller_params = yaml.safe_load(_can_opener)
        self.K = torch.tensor(_controller_params['K'], dtype=torch.float32,     device="cuda")
        self.A_d = torch.tensor(_controller_params['A_d'], dtype=torch.float32, device="cuda")
        self.B_d = torch.tensor(_controller_params['B_d'], dtype=torch.float32, device="cuda")

        self.Q = torch.eye(12, dtype=torch.float32, device="cuda") * 1e-3
        self.R = torch.eye(12, dtype=torch.float32, device="cuda") * 1e-3

    def update(self, state, u):
        _u = u.view(-1, 1)
        self.state_e = self.A_d @ self.state_p + self.B_d @ _u
        self.P = self.A_d @ self.P @ self.A_d.t() + self.Q
        self.K  = self.P @ self.H.t() @ torch.inverse(self.H @ self.P @ self.H.t() + self.R)
        self.state_p = self.state_e + self.K @ (state - self.H @ self.state_e)
        self.P = (torch.eye(12, dtype=torch.float32, device="cuda") - self.K @ self.H) @ self.P


class Kxontroller:
    time_summer : float = 0.0
    lqr_status: float = 0.0
    move_status: float = 0.0
    u_eq: torch.tensor = torch.tensor([7.43987615, 7.33890313, 7.06352215, 7.16449517], dtype=torch.float32, device="cuda")

    def __init__(self, num_envs, device="cuda", freq=100):
        _can_opener = open("controller.yaml", "r")
        _controller_params = yaml.safe_load(_can_opener)

        self.thrust = torch.zeros([num_envs, 4], dtype=torch.float32, device=device)
        self.ve_thrust = torch.zeros([num_envs, 4, 3], dtype=torch.float32, device=device)
        self.ve_moment = torch.zeros([num_envs, 4, 3], dtype=torch.float32, device=device)

        self.K = torch.tensor(_controller_params['K_LQR'], dtype=torch.float32, device=device)
        self.P = torch.tensor(_controller_params['P_LQR'], dtype=torch.float32, device=device)

        self.K_LQR = torch.tensor(_controller_params['K_LQR'], dtype=torch.float32, device=device)
        self.P_LQR = torch.tensor(_controller_params['P_LQR'], dtype=torch.float32, device=device)


        self.K_MOVE = torch.tensor(_controller_params['K_MOVE'], dtype=torch.float32, device=device)
        self.P_MOVE = torch.tensor(_controller_params['P_MOVE'], dtype=torch.float32, device=device)

        self.kilter = Kilter()

        self.control_dt = 1.0 / freq
        self.time_summer = 0.0
        self.device = device

    def step(self, desired_states, states, dt):
        self.time_summer += dt
        if self.time_summer >= self.control_dt:
            self.time_summer = 0.0

            for env, sp, thrust in zip(states, desired_states, self.thrust):
                _temp_out = self.update(
                        sp, 
                        env[0][0:3], 
                        env[0][3:6], 
                        env[0][6:9], 
                        env[0][9:12],
                        thrust
                        )

                thrust.copy_(_temp_out)
            
        else:
            self.thrust = self.thrust


        self.ve_thrust[:, :, 2] = self.thrust
        self.ve_moment[:, :, 2] = self.thrust * 0.0098
        self.ve_moment[:, 1, 1] *= -1
        self.ve_moment[:, 3, 1] *= -1

    def update(self, desired_pos, current_pos, current_vel, current_att, current_ang_vel, thrust):

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device).float()
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(self.device).float()
            elif isinstance(x, (float, int)):
                return torch.tensor([x], dtype=torch.float32, device=self.device)
            elif isinstance(x, (list, tuple)):
                return torch.tensor(x, dtype=torch.float32, device=self.device)
            else:
                raise TypeError(f"Unsupported type for tensor conversion: {type(x)}")


        desired_pos = to_tensor(desired_pos).flatten()
        current_pos = to_tensor(current_pos).flatten()
        current_vel = to_tensor(current_vel).flatten()
        current_att = to_tensor(current_att).flatten()
        current_ang_vel = to_tensor(current_ang_vel).flatten()


        _state = torch.cat(
                (
                    current_pos,
                    current_vel,
                    current_att,
                    current_ang_vel
                    )
                )

        self.kilter.update(_state, thrust)

        _p_state_offset = torch.cat(
                (
                    (current_pos - desired_pos),
                    current_vel,
                    current_att,
                    current_ang_vel
                    )
                )

        _state_offset = torch.cat(
                (
                    (current_pos - desired_pos),
                    current_vel,
                    current_att,
                    current_ang_vel
                    )
                )

        current_state = to_tensor(_state).flatten()
        current_state_offset = to_tensor(_state_offset).flatten()
        current_p_state_offset = to_tensor(_p_state_offset).flatten()
        print(current_p_state_offset)
        out = self.decide(current_p_state_offset)
        return out

    def decide(self, current_state_offset):
        self.lqr_status = current_state_offset @ self.P_LQR @ current_state_offset.t() 
        self.move_status = current_state_offset @ self.P_MOVE @ current_state_offset.t() 
        if abs(self.lqr_status) <= 1:
            output = self.u_eq + torch.matmul(self.K_LQR, -current_state_offset)
        elif abs(self.move_status) <= 1:
            output = self.u_eq + torch.matmul(self.K_MOVE, -current_state_offset)
        else:
            output = self.u_eq + torch.matmul(self.K, -current_state_offset)

        return output
