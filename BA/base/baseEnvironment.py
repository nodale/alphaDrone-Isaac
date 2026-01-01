from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

#class EnvironmentWindow(BaseEnvWindow):
#    def __init__(self, env: DroneRunner, window_name: str = "IsaacLab"):
#        super().__init__(env, window_name)
#        with self.ui_window_elements["main_vstack"]:
#            with self.ui_window_elements["debug_frame"]:
#                with self.ui_window_elements["debug_vstack"]:
#                    self._create_debug_vis_ui_element("targets", self.env)


from baseDrone import Drone

@configclass
class DroneEnvironment(DirectRLEnvCfg):
    episode_length_s = 120.0
    decimation = 1
    action_space = 4
    observation_space = 6
    state_space = 0
    debug_vis = False
    num_envs = 4

#    ui_window_class_type = EnvironmentWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2, 
        env_spacing=2.5, 
        replicate_physics=True
    )

    def __init__(self):
        self.drone = Drone(self.num_envs)
        self.drone.articulation_cfg.replace(prim_path="/World/envs/env_.*/drone")
