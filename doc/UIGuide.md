This is for version 4.2.0

Middle mouse button to move perspective around
Right mouse button to rotate camera

To add a ground:
Create -> Physics -> Ground Plane

To add gravity and collision:
click object -> +Add -> Rigid Body with Colliders Preset

To import URDF:
launch Interface -> Isaac Utils -> Workflows -> URDF Importer -> scroll down -> Input

or using pythons script:

```python
from omni.importer.urdf import _urdf

urdf_interface = _urdf.acquire_urdf_interface()
config = _urdf.ImportConfig()
config.fix_base = True
config.create_physics_scene = True
config.collision_from_visuals = False
config.self_collision = False
config.make_instanceable = True

urdf_interface.import_urdf("path to the URDF file", config)

from pxr import Usd

stage = Usd.Stage.Open("the importedURDF's usd file")
prim = stage.GetPrimAtPath("path to the usd file")
prim.SetInstanceable(True)
stage.GetRootLayer().Save()

from pxr import UsdPhysics

UsdPhysics.RigidBodyAPI.Apply(prim)
UsdPhysics.CollisionAPI.Apply(prim)


```

After importing, do:
[x] Instancable -> add gravity and collision
