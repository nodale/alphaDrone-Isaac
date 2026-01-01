import torch as torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

class Drone():
    state : torch.tensor
    thrust : torch.tensor

    articulation_cfg : ArticulationCfg = ArticulationCfg(
                spawn=sim_utils.UsdFileCfg(usd_path="drone/test9.usda"),
                prim_path="/World/envs/env_.*/drone",
                actuators={"rotors": ImplicitActuatorCfg(joint_names_expr=["rotor_[1-4]_joint"], damping=None, stiffness=None)},
                init_state=ArticulationCfg.InitialStateCfg(pos=[0.0, 0.0, 0.2])
                )   

    def __init__(self, num_envs, device="cuda"):
        self.state = torch.zeros(num_envs, 12, device=device)
        self.thrust = torch.zeros(num_envs, 4, device=device)
        self.moment = torch.zeros(num_envs, 4, device=device)

        self.setpoint = torch.zeros(num_envs, 3, device=device)

        #self.articulation_cfg.replace(prim_path="/World/envs/env_.*/drone")
        #self.articulation = Articulation(self.articulation_cfg)

    def init(self):
        self.rotor_id = self.articulation.find_bodies("rotor_[1-4]")

        
