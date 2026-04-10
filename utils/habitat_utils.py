import json
from collections import defaultdict
from glob import glob
from typing import Any, Dict

import cv2
import habitat_sim
from PIL import Image
import numpy as np

def get_test_split_files(env='gibson'):
    dirs = glob('/home/faith/GitRepos/habitat/' + env.lower() + '_test_splits/*/*.json')
    env_file_list = defaultdict(list)
    for d in dirs:
        with open(d) as f:
            files = json.load(f)
        key = f.name.split('splits/')[-1].split('.')[0]
        for e in files['episodes']:
            env_file_list[key].append(e['scene_id'].split('/')[-1])

    temp = []
    for key in env_file_list.keys():
        env_file_list[key] = set(env_file_list[key])
        temp.extend(env_file_list[key])
    env_file_list['all'] = set(temp)

    return env_file_list


def saveOBS(obs, folder_path, i):
    i_str = str(i)
    while len(i_str) < 5: i_str = '0' + i_str
    rgb = obs['color_sensor']
    depth = obs['depth_sensor']
    rgb_path = folder_path + '/rgb_' + i_str + '.jpg' # '.png'
    depth_path = folder_path + '/depth_' + i_str + '.jpg' # '.png'
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_path, depth)

    # depth v2
    # depth from the official website codebase >>>
    if depth.size != 0:
        depth_v2 = Image.fromarray((depth / 10 * 255).astype(np.uint8), mode="L")
        # depth_img = Image.fromarray(depth);
        depth_v2_path = folder_path + '/depth_v2_' + str(i) + '.jpg' # '.png'
        depth_v2.save(depth_v2_path)
    # depth from the official website codebase <<<
    return [rgb_path, depth_path]  # returning names so can add to nodes

def saveRealOBS(obs, folder_path, i):
    i_str = str(i)
    while len(i_str) < 5: i_str = '0' + i_str
    rgb = obs
    rgb_path = folder_path + '/rgb_' + i_str + '.jpg' # '.png'
    cv2.imwrite(rgb_path, rgb)

# build SimulatorConfiguration
def make_cfg(settings: Dict[str, Any]):
    r"""Isolates the boilerplate code to create a habitat_sim.Configuration from a settings dictionary.

    :param settings: A dict with pre-defined keys, each a basic simulator initialization parameter.

    Allows configuration of dataset and scene, visual sensor parameters, and basic agent parameters.

    Optionally creates up to one of each of a variety of aligned visual sensors under Agent 0.

    The output can be passed directly into habitat_sim.simulator.Simulator constructor or reconfigure to initialize a Simulator instance.
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    if "scene_dataset_config_file" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    sim_cfg.frustum_culling = settings.get("frustum_culling", False)
    if "enable_physics" in settings:
        sim_cfg.enable_physics = settings["enable_physics"]
    if "physics_config_file" in settings:
        sim_cfg.physics_config_file = settings["physics_config_file"]
    if "scene_light_setup" in settings:
        sim_cfg.scene_light_setup = settings["scene_light_setup"]
    sim_cfg.gpu_device_id = 0
    if not hasattr(sim_cfg, "scene_id"):
        raise RuntimeError(
            "Error: Please upgrade habitat-sim. SimulatorConfig API version mismatch"
        )
    sim_cfg.scene_id = settings["scene"]

    # define default sensor parameters (see src/esp/Sensor/Sensor.h)
    sensor_specs = []

    def create_camera_spec(**kw_args):
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.resolution = [settings["height"], settings["width"]]
        camera_sensor_spec.position = [0, settings["sensor_height"], 0]
        for k in kw_args:
            setattr(camera_sensor_spec, k, kw_args[k])
        return camera_sensor_spec

    if settings["color_sensor"]:
        color_sensor_spec = create_camera_spec(
            uuid="color_sensor",
            hfov=settings["hfov"],
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(color_sensor_spec)

    if settings["depth_sensor"]:
        depth_sensor_spec = create_camera_spec(
            uuid="depth_sensor",
            hfov=settings["hfov"],
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(depth_sensor_spec)

    if settings["semantic_sensor"]:
        semantic_sensor_spec = create_camera_spec(
            uuid="semantic_sensor",
            hfov=settings["hfov"],
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(semantic_sensor_spec)

    if settings["ortho_rgba_sensor"]:
        ortho_rgba_sensor_spec = create_camera_spec(
            uuid="ortho_rgba_sensor",
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
        )
        sensor_specs.append(ortho_rgba_sensor_spec)

    if settings["ortho_depth_sensor"]:
        ortho_depth_sensor_spec = create_camera_spec(
            uuid="ortho_depth_sensor",
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
        )
        sensor_specs.append(ortho_depth_sensor_spec)

    if settings["ortho_semantic_sensor"]:
        ortho_semantic_sensor_spec = create_camera_spec(
            uuid="ortho_semantic_sensor",
            far=settings["zfar"],
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
        )
        sensor_specs.append(ortho_semantic_sensor_spec)

    # TODO Figure out how to implement copying of specs
    def create_fisheye_spec(**kw_args):
        fisheye_sensor_spec = habitat_sim.FisheyeSensorDoubleSphereSpec()
        fisheye_sensor_spec.uuid = "fisheye_sensor"
        fisheye_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        fisheye_sensor_spec.sensor_model_type = (
            habitat_sim.FisheyeSensorModelType.DOUBLE_SPHERE
        )

        # The default value (alpha, xi) is set to match the lens "GoPro" found in Table 3 of this paper:
        # Vladyslav Usenko, Nikolaus Demmel and Daniel Cremers: The Double Sphere
        # Camera Model, The International Conference on 3D Vision (3DV), 2018
        # You can find the intrinsic parameters for the other lenses in the same table as well.
        fisheye_sensor_spec.xi = -0.27
        fisheye_sensor_spec.alpha = 0.57
        fisheye_sensor_spec.focal_length = [364.84, 364.86]

        fisheye_sensor_spec.resolution = [settings["height"], settings["width"]]
        # The default principal_point_offset is the middle of the image
        fisheye_sensor_spec.principal_point_offset = None
        # default: fisheye_sensor_spec.principal_point_offset = [i/2 for i in fisheye_sensor_spec.resolution]
        fisheye_sensor_spec.position = [0, settings["sensor_height"], 0]
        for k in kw_args:
            setattr(fisheye_sensor_spec, k, kw_args[k])
        return fisheye_sensor_spec

    if settings["fisheye_rgba_sensor"]:
        fisheye_rgba_sensor_spec = create_fisheye_spec(uuid="fisheye_rgba_sensor")
        sensor_specs.append(fisheye_rgba_sensor_spec)
    if settings["fisheye_depth_sensor"]:
        fisheye_depth_sensor_spec = create_fisheye_spec(
            uuid="fisheye_depth_sensor",
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
        )
        sensor_specs.append(fisheye_depth_sensor_spec)
    if settings["fisheye_semantic_sensor"]:
        fisheye_semantic_sensor_spec = create_fisheye_spec(
            uuid="fisheye_semantic_sensor",
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
        )
        sensor_specs.append(fisheye_semantic_sensor_spec)

    def create_equirect_spec(**kw_args):
        equirect_sensor_spec = habitat_sim.EquirectangularSensorSpec()
        equirect_sensor_spec.uuid = "equirect_rgba_sensor"
        equirect_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        equirect_sensor_spec.resolution = [settings["height"], settings["width"]]
        equirect_sensor_spec.position = [0, settings["sensor_height"], 0]
        for k in kw_args:
            setattr(equirect_sensor_spec, k, kw_args[k])
        return equirect_sensor_spec

    if settings["equirect_rgba_sensor"]:
        equirect_rgba_sensor_spec = create_equirect_spec(uuid="equirect_rgba_sensor")
        sensor_specs.append(equirect_rgba_sensor_spec)

    if settings["equirect_depth_sensor"]:
        equirect_depth_sensor_spec = create_equirect_spec(
            uuid="equirect_depth_sensor",
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
        )
        sensor_specs.append(equirect_depth_sensor_spec)

    if settings["equirect_semantic_sensor"]:
        equirect_semantic_sensor_spec = create_equirect_spec(
            uuid="equirect_semantic_sensor",
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
        )
        sensor_specs.append(equirect_semantic_sensor_spec)

    # create agent specifications
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "stop": habitat_sim.agent.ActionSpec(
            "stop"
        ),
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),  # TODO: add noisy actions here???
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])
