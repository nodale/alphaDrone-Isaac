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
    def __init__(self):
        self.episode_length_s = 120.0
        self.decimation = 1
        self.action_space = 4
        self.observation_space = 6
        self.state_space = 0
        self.debug_vis = False

    #    ui_window_class_type = EnvironmentWindow

        self.sim = SimulationCfg(
            dt=1 / 100,
            render_interval=self.decimation,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.0,
            ),
        )

        self.terrain = TerrainImporterCfg(
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

        self.scene = InteractiveSceneCfg(
            num_envs=2, 
            env_spacing=2.5, 
            replicate_physics=True
        )

        self.drone = Drone(2)
        self.drone.articulation_cfg.replace(prim_path="/World/envs/env_.*/drone")
