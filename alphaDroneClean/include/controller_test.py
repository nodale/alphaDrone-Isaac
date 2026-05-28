from Kxontroller import Kxontroller

import torch

setpoint = torch.tensor([
    0.0, 0.0, 0.0,
    ], device="cuda")
setpoint = setpoint.unsqueeze(0)


obs = torch.tensor([
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    ], device="cuda")
obs = obs.unsqueeze(0)

controller = Kxontroller(num_envs=1)
controller.step(desired_pos=setpoint, states=obs, dt=1.0)

print(controller.thrust)


