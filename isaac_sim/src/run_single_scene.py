
import sys
import os
import json
import csv
from datetime import datetime
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.prims import RigidPrim, GeometryPrim

from pxr import Sdf, UsdLux, UsdGeom
import omni.appwindow
import omni.usd
import omni.replicator.core as rep
from omni.isaac.core.articulations import Articulation

scene_idx = int(sys.argv[1])
with open("/root/Documents/spot_63_scene_config.json", "r") as f:
    scenes = json.load(f)

scene = scenes[scene_idx]
props_config = scene["props"]
destination_label = scene["destination"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/isaac-sim/recordings/scene_{scene_idx:02d}"
os.makedirs(output_dir, exist_ok=True)

first_step = True
reset_needed = False
base_command = np.zeros(3)
i = 0

def on_physics_step(step_size):
    global first_step, reset_needed
    if first_step:
        spot.initialize()
        first_step = False
    elif reset_needed:
        my_world.reset(True)
        reset_needed = False
        first_step = True
    else:
        spot.forward(step_size, base_command)

my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 50)
assets_root_path = get_assets_root_path()

# Load environment
define_prim("/World/Ground", "Xform")
stage = omni.usd.get_context().get_stage()
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
stage.GetPrimAtPath("/World/Ground").GetReferences().AddReference(asset_path)

# Lighting
old_light_path = Sdf.Path("/World/Ground/SphereLight")
old_light = stage.GetPrimAtPath(old_light_path)
if old_light.IsValid():
    old_light.SetActive(False)
light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Ground/DistantLight"))
light.CreateIntensityAttr(3000.0)
light.CreateAngleAttr(5.0)

# Robot
spot = SpotFlatTerrainPolicy(
    prim_path="/World/Spot",
    name="Spot",
    position=np.array([0, 0, 0.8]),
)

# Camera
camera_path = "/World/Spot/body/Camera"
define_prim(camera_path, "Xform")
camera_usd_path = assets_root_path + "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
get_prim_at_path(camera_path).GetReferences().AddReference(camera_usd_path)
cam_xform = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
cam_xform.AddTranslateOp().Set((0.45, 0.0, 0.0))
cam_xform.AddRotateXYZOp().Set((0.0, 24.0, 0.0))

# Props
for idx, prop in enumerate(props_config):
    path = f"/World/Prop_{idx}"
    define_prim(path, "Xform")
    get_prim_at_path(path).GetReferences().AddReference(assets_root_path + prop["path"])
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xform.AddTranslateOp().Set(tuple(prop["translation"]))
    xform.AddRotateXYZOp().Set(tuple(prop["rotation"]))
    RigidPrim(path)
    GeometryPrim(path).apply_collision_apis()

# Replicator setup
cam_prim_path = "/World/Spot/body/Camera/RSD455/Camera_OmniVision_OV9782_Color"
render_product = rep.create.render_product(cam_prim_path, resolution=(1280, 720))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=output_dir, rgb=True, semantic_segmentation=False)
writer.attach([render_product])

# CSV file for saving commands
csv_path = os.path.join(output_dir, "commands.csv")
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["time", "vx", "vy", "omega"])

# CSV file for saving robot velocity
vel_csv_path = os.path.join(output_dir, "velocity.csv")
vel_csv_file = open(vel_csv_path, mode="w", newline="")
vel_csv_writer = csv.writer(vel_csv_file)
vel_csv_writer.writerow(["time", "vx", "vy", "vz", "wx", "wy", "wz"])  # header

# Simulation
my_world.reset()
my_world.add_physics_callback("physics_step", on_physics_step)
start_time = my_world.current_time
duration = 5
spot_articulation = Articulation(prim_path="/World/Spot")


while simulation_app.is_running() and my_world.current_time - start_time < duration:
    my_world.step(render=True)
    if my_world.is_stopped():
        reset_needed = True

    if my_world.is_playing():
        if destination_label == "left":
            if 50 <= i < 70:
                base_command[:] = [0, 0, 1]
            elif 70 <= i < 180:
                base_command[:] = [1, 0, 0]
            else:
                base_command[:] = [0, 0, 0]
        elif destination_label == "straight":
            if 50 <= i < 140:
                base_command[:] = [1, 0, 0]
            else:
                base_command[:] = [0, 0, 0]
        elif destination_label == "right":
            if 50 <= i < 75:
                base_command[:] = [0, 0, -1]
            elif 75 <= i < 180:
                base_command[:] = [1, 0, 0]
            else:
                base_command[:] = [0, 0, 0]
        else:
            base_command[:] = [0, 0, 0]

        csv_writer.writerow([my_world.current_time] + base_command.tolist())

        lin_vel = spot_articulation.get_linear_velocity()
        ang_vel = spot_articulation.get_angular_velocity()
        vel_csv_writer.writerow([
            my_world.current_time,
            *lin_vel.tolist(),  # vx, vy, vz
            *ang_vel.tolist()   # wx, wy, wz
        ])

        i += 1

csv_file.close()
vel_csv_file.close()
simulation_app.close()
