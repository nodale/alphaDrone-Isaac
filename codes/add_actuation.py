from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path

from isaacsim.core.prims import GeometryPrim, RigidPrim

import os
os.chdir("/home/azapata/JoeysStuffs/alphaDrone-Isaac/URDF_template")

def apply_rotor_forces(rotors):
    for i in range(rotors.count):
<<<<<<< HEAD
        rotors.apply_forces(
            indices=[i],
            forces=np.array([0, 10, 0]),
=======
        rotors.apply_forces_and_torques_at_pos(
            indices=[i],
            forces=np.array([0, 8, 0]),
            torques=np.array([0, 100*(-1)**i, 0]),
>>>>>>> 1539824 (updated for torques)
            is_global=False
        )

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

assets_root_path = "/home/azapata/JoeysStuffs/alphaDrone-Isaac/URDF_template"
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

asset_path = assets_root_path + "test.usda"
robot1 = add_reference_to_stage(usd_path="test.usda", prim_path="/World/drone")

rigid_prim = RigidPrim(prim_paths_expr="/World/drone/rotor_[1-4]")
my_world.scene.add(rigid_prim)

for i in range(5):
    print("resetting...")
    my_world.reset()
    for j in range(500):
        apply_rotor_forces(rigid_prim)
        my_world.step(render=True)
    if args.test is True:
        break
simulation_app.close()


