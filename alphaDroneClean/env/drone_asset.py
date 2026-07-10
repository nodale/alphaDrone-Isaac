import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ImuCfg

from control.lqr import Kxontroller


class Drone:
    drone_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/drone",
        spawn=sim_utils.UsdFileCfg(usd_path="assets/fly_boy/fly_boy.usda"),
        actuators={"rotors": ImplicitActuatorCfg(
            joint_names_expr=["rotor_[1-4]_joint"], damping=None, stiffness=None
        )},
        init_state=ArticulationCfg.InitialStateCfg(pos=[0.0, 0.0, 0.185]),
    )

    imu_cfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/drone/base_link",
        update_period=1.0 / 800.0,
        history_length=2,
        debug_vis=False,
    )

    def __init__(self, num_envs, device="cuda"):
        self.esc_delay = 1.0 / 400.0

        self.thrust = torch.zeros(num_envs, 4, device=device)
        self.moment = torch.zeros(num_envs, 4, device=device)
        self.thrust_noise = torch.zeros(num_envs, 4, 3, device=device)
        self.moment_noise = torch.zeros(num_envs, 4, 3, device=device)

        self.controller = Kxontroller(num_envs=num_envs, esc_delay=self.esc_delay)
        self.rotor_ids = torch.zeros(num_envs, 4, device=device)
        self.setpoint = torch.zeros(num_envs, 3, device=device)
