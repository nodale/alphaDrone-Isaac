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
import os
os.chdir("/home/azapata/JoeysStuffs/alphaDrone-Isaac/URDF_template")

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
robot1 = add_reference_to_stage(usd_path="test.usda", prim_path="/World/dron")

for i in range(5):
    print("resetting...")
    my_world.reset()
    for j in range(500):
        my_world.step(render=True)
    if args.test is True:
        break
simulation_app.close()


