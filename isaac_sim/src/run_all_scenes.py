import subprocess

for scene_idx in range(63):
    print(f"Running scene {scene_idx}...")
    result = subprocess.run(
        ["./python.sh", "/root/Documents/run_single_scene.py", str(scene_idx)],
        cwd="/isaac-sim",
    )
    if result.returncode != 0:
        print(f"Scene {scene_idx} failed with return code {result.returncode}")
        break
