import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.lqr import Kxontroller
import torch

setpoint = torch.tensor([[0.0, 0.0, 0.0]], device="cuda")
obs = torch.zeros(1, 13, device="cuda")

controller = Kxontroller(num_envs=1)
controller.step(desired_pos=setpoint, states=obs, dt=1.0)

print(controller.thrust)
