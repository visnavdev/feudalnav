import argparse
import os
import random
from glob import glob

import cv2
import habitat_sim
import numpy as np
import torch
from matplotlib import pyplot as plt

from models import Worker, HighManager
from utils.habitat_utils import make_cfg


def getArgs():
    parser = argparse.ArgumentParser()

    # Superglue arguments
    parser.add_argument('--nms_radius', type=int, default=4,
                        help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument('--match_threshold', type=float, default=0.2, help='SuperGlue match threshold')
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='indoor', help='SuperGlue weights')
    parser.add_argument('--max_keypoints', type=int, default=1024,
                        help='Maximum number of keypoints detected by Superpoint (\'-1\' keeps all keypoints)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.01,
                        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--x_size', type=int, default=640, help='x dimension of the image')
    parser.add_argument('--y_size', type=int, default=480, help='y dimension of the image')

    # Isomap Args
    parser.add_argument('--n_neighbors', type=int, default=1,
                        help='number of nearest neighbors to consider when clustering')

    # Habitat arguments
    parser.add_argument('--meters_per_pixel', type=float, default=0.1,
                        help='meters per pixel for topdown pathfinder map')
    parser.add_argument('--habitat_path', type=str, default='/home/faith/GitRepos/habitat/',
                        help='path to the folder with the habitat data')
    parser.add_argument('--data_path', type=str, default='/home/faith/Desktop/Red_Disk_Data/habitat_graph_data/',
                        help='path to the folder with the dataset graph data')
    parser.add_argument('--dataset', type=str, default='gibson', choices=['gibson', 'matterport'],
                        help='which dataset to use')
    parser.add_argument('--depth_threshold', type=float, default=0.4,
                        help='how close the agent is allowed to get to objects in the sim in meters')

    return parser.parse_args()


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global lRefPt, rRefPt
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        lRefPt = [x, y]
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_RBUTTONDOWN:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        rRefPt = [x, y]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = getArgs()
    rRefPt = []
    lRefPt = []
    landmarks = []

    # initialize environment with random
    sim = None
    while sim is None:
        try:
            test_scene_file = random.choice(glob(args.habitat_path + args.dataset + '/*'))
            sim_settings = {"scene_dataset_config_file": "default",
                            "scene": test_scene_file,
                            "default_agent": 0,
                            "sensor_height": 1.5,
                            "width": args.x_size,
                            "height": args.y_size,
                            "hfov": 120,
                            "zfar": 1000.0,
                            "color_sensor": True,
                            "semantic_sensor": False,
                            "depth_sensor": True,
                            "ortho_rgba_sensor": False,
                            "ortho_depth_sensor": False,
                            "ortho_semantic_sensor": False,
                            "fisheye_rgba_sensor": False,
                            "fisheye_depth_sensor": False,
                            "fisheye_semantic_sensor": False,
                            "equirect_rgba_sensor": False,
                            "equirect_depth_sensor": False,
                            "equirect_semantic_sensor": False,
                            "seed": 1,
                            "physics_config_file": "data/default.physics_config.json",
                            "enable_physics": False}
            habitat_cfg = make_cfg(sim_settings)
            sim = habitat_sim.Simulator(habitat_cfg)
        except:
            pass

    # some useful stuff for later?
    # sim.get_active_scene_graph()
    # sim.get_active_semantic_scene_graph()
    # sim.navmesh_visualization

    # initializing the agent in the environment
    agent = sim.get_agent(0)
    agent_position = sim.pathfinder.get_random_navigable_point()
    agent_state = habitat_sim.AgentState()
    agent_state.position = agent_position
    agent.set_state(agent_state)
    # agent_state = agent.get_state()
    # agent_location = agent_state.position
    # agent_orient = agent_state.rotation

    worker = Worker(args, device)
    highManager = HighManager(args)
    if os.path.exists('test_imgs/'):
        worker.reset('test_imgs/')

    if not os.path.exists('test_imgs/'):
        os.mkdir('test_imgs/')

    # get initial observation
    obs = sim.step("turn_right")

    # run through loop until done
    cv2.namedWindow("Sim World")
    cv2.setMouseCallback("Sim World", click_and_crop)
    while (1):
        cv2.imshow("Sim World", obs['color_sensor'])
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):  # q key to stop
            break

        # if user click and dragged to get a section of the image
        if len(lRefPt) == 2:
            goal_img = np.array(lRefPt)
            # goal_img = obs[refPt[0][0]:refPt[1][0], refPt[0][1]:refPt[1][1], :]
            # goal_img = cv2.cvtColor(goal_img, cv2.COLOR_RGB2GRAY)
            obs = worker.step(obs, goal_img, sim, agent, lRefPt)
            lRefPt = []
        if len(rRefPt) == 2:
            worker.graph.addLandmark([rRefPt, worker.graph.current_node])
            rRefPt = []
            print('Landmarks:', worker.graph.landmarks)
            if len(worker.graph.landmarks) > args.n_neighbors:
                manifold_map = highManager.updateManifoldMap(worker.graph.getLandmarkGraph())
                plt.clf()
                plt.scatter(manifold_map[:, 0], manifold_map[:, 1])
                plt.title("Current Manifold Map")
                plt.pause(1)

    worker.graph.save('test_imgs/worker_graph.json')
    sim.close()
