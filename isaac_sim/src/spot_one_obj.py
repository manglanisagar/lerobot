from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.prims import RigidPrim, GeometryPrim

from pxr import Sdf, UsdLux, UsdGeom
import omni.appwindow  # Contains handle to keyboard
import omni.usd
import omni.replicator.core as rep

import os
from datetime import datetime
import csv

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/isaac-sim/recordings/output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

first_step = True
reset_needed = False

# initialize robot on first step, run robot advance
def on_physics_step(step_size) -> None:
    global first_step
    global reset_needed
    if first_step:
        spot.initialize()
        first_step = False
    elif reset_needed:
        my_world.reset(True)
        reset_needed = False
        first_step = True
    else:
        spot.forward(step_size, base_command)


# spawn world
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 50)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# spawn warehouse scene
prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
prim.GetReferences().AddReference(asset_path)

# Add distant light
stage = omni.usd.get_context().get_stage()
old_light_path = Sdf.Path("/World/Ground/SphereLight")
old_light = stage.GetPrimAtPath(old_light_path)
old_light.SetActive(False)
distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Ground/DistantLight"))
distant_light.CreateIntensityAttr(3000.0)
distant_light.CreateAngleAttr(5.0)

# spawn robot
spot = SpotFlatTerrainPolicy(
    prim_path="/World/Spot",
    name="Spot",
    position=np.array([0, 0, 0.8]),
)


# Define prim path for the camera under the robot body
camera_path = "/World/Spot/body/Camera"
define_prim(camera_path, "Xform")  # create if doesn't exist

# Reference the D455 camera USD asset
camera_usd_path = assets_root_path + "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
camera_prim = get_prim_at_path(camera_path)
camera_prim.GetReferences().AddReference(camera_usd_path)

# Set the transform relative to /World/Spot/body
camera_xform = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
camera_xform.AddTranslateOp().Set((0.45, 0.0, 0.0))     # 45 cm forward
camera_xform.AddRotateXYZOp().Set((0.0, 24.0, 0.0))      # 24 degrees yaw


# Define the prim path for the object
object_path = "/World/object"
define_prim(object_path, "Xform")

# Reference the USD asset
object_usd_path = assets_root_path + "/Isaac/Props/YCB/Axis_Aligned/011_banana.usd"
object_prim = get_prim_at_path(object_path)
object_prim.GetReferences().AddReference(object_usd_path)

# Set object's position and orientation in the world
stage = omni.usd.get_context().get_stage()
object_xform = UsdGeom.Xformable(stage.GetPrimAtPath(object_path))
object_xform.AddTranslateOp().Set((3.0, 0.0, 0.1))     # Place it in front-right of robot
object_xform.AddRotateXYZOp().Set((0.0, 0.0, 0.0))      # No rotation

# Make object a rigid body so it responds to gravity
RigidPrim(object_path)

# Add collision API so it can collide with other objects
object_geom = GeometryPrim(object_path)
object_geom.apply_collision_apis()


# Use existing camera prim path
camera_prim_path = "/World/Spot/body/Camera/RSD455/Camera_OmniVision_OV9782_Color"

# Create a render product from the existing camera
render_product = rep.create.render_product(camera_prim_path, resolution=(1280, 720))

# Create and initialize a writer to save RGB images
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=output_dir,
    rgb=True,
    semantic_segmentation=False,
)

# Attach writer to the render product
writer.attach([render_product])

# Open CSV file in the output folder
csv_path = os.path.join(output_dir, "commands.csv")
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["time", "vx", "vy", "omega"])  # header


# Starting world
my_world.reset()
my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

# robot command
base_command = np.zeros(3)

i = 0
start_time = my_world.current_time
duration = 4  # seconds
while simulation_app.is_running() and my_world.current_time - start_time < duration:
    my_world.step(render=True)

    if my_world.is_stopped():
        reset_needed = True
    if my_world.is_playing():
        if i >= 50 and i < 140:
            base_command = np.array([1, 0, 0])
        else:
            base_command = np.array([0, 0, 0])
        i += 1
        csv_writer.writerow([my_world.current_time] + base_command.tolist())

csv_file.close()
simulation_app.close()
