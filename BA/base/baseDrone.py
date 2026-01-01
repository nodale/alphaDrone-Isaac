import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

class Drone():
    articulation_cfg : ArticulationCfg = ArticulationCfg(
                spawn=sim_utils.UsdFileCfg(usd_path="drone/test9.usda"),
                prim_path="/World/envs/env_.*/drone",
                actuators={"rotors": ImplicitActuatorCfg(joint_names_expr=["rotor_[1-4]_joint"], damping=None, stiffness=None)},
                init_state=ArticulationCfg.InitialStateCfg(pos=[0.0, 0.0, 0.2])
                )   

