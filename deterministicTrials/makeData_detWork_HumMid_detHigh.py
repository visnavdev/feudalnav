'''
Usage:

python deterministicTrials/makeData_detWork_HumMid_detHigh.py --habitat_path /home/brcao/Data/datasets/habitat/gs --env_name Neibert

python deterministicTrials/makeData_detWork_HumMid_detHigh.py --absolute_scene_path \
    /home/brcao/Data/datasets/habitat/mp/HM3Dv0.2/untarred/train/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb
    
python deterministicTrials/makeData_detWork_HumMid_detHigh.py --absolute_scene_path \
 /home/brcao/Data/datasets/habitat/mp/HM3Dv0.2/untarred/train/00547-9h5JJxM6E5S/9h5JJxM6E5S.basis.glb
'''

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

    # Data collection arguments
    parser.add_argument('--action_limit', type=int, default=500, help='max number of environment steps to make')

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
    parser.add_argument('-ds', '--dataset', type=str, default='gibson', choices=['gs', 'gibson', 'mp', 'matterport'],
                        help='which dataset to use')
    parser.add_argument('-dst', '--dataset_type', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--env_name', type=str, default='None', help='env name in glb format, e.g. env_name.glb')
    parser.add_argument('--depth_threshold', type=float, default=0.4,
                        help='how close the agent is allowed to get to objects in the sim in meters')
    parser.add_argument('--letter', type=str, default=None, help='choose environments starting with this letter')
    parser.add_argument('--save_root', type=str, default='./human_click_dataset', help='Save data root')
    parser.add_argument('--absolute_scene_path', type=str, default='unspecified')
    parser.add_argument('-dm', '--display_map', action='store_true')
    parser.add_argument('-sen', '--save_env_name', action='store_true')
    parser.add_argument('-ms', '--max_steps', type=int, default=500)

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
    
    if True:
        if args.absolute_scene_path == 'unspecified':
            if args.dataset == 'gs' or args.dataset == 'gibson':
                env_file = args.habitat_path + '/' + args.dataset + '/' + args.env_name + '.glb'
            elif args.dataset == 'mp' or args.dataset == 'matterport':
                env_file = args.habitat_path + '/' + args.dataset + '/HM3Dv0.2/untarred/' + \
                    args.dataset_type + '/' + args.env_name + '/' + args.env_name[6:] + '.basis.glb'
        else: env_file = args.absolute_scene_path
        print('\n env_file: ', env_file)

        try:
            
            sim_settings = {"scene_dataset_config_file": "default",
                            "scene": env_file,
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
            sim = None

        if sim is not None:
            rRefPt = []
            lRefPt = []

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
            # print('\n dir(agent): ', dir(agent))
            # print('\n agent.agent_config: ', agent.agent_config)

            worker = Worker(args, device, sim.curr_scene_name)
            highManager = HighManager(args)
            if not os.path.exists(args.save_root): os.mkdir(args.save_root)

            # make the folders for each trajectory predicted
            existing_traj_folders = glob(args.save_root + '/*')
            if args.save_env_name:
                save_path = args.save_root + '/traj_' + args.env_name + '/'
            else:
                most_recent_traj = str(len(existing_traj_folders))
                while len(most_recent_traj) < 5: most_recent_traj = '0' + most_recent_traj
                save_path = args.save_root + '/traj_' + most_recent_traj + '/'
                print('\n save_path: ', save_path)
            os.mkdir(save_path)

            # get initial observation
            obs = sim.step("turn_right")

            # run through loop until done
            cv2.namedWindow("Sim World")
            cv2.setMouseCallback("Sim World", click_and_crop)
            steps = 0
            while steps < args.max_steps:
                cv2.imshow("Sim World", obs['color_sensor'])
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q") or len(worker.graph.actions) >= args.action_limit:  # q key to stop
                    break

                # if user click and dragged to get a section of the image
                if len(lRefPt) == 2:
                    goal_img = np.array(lRefPt)
                    # goal_img = obs[refPt[0][0]:refPt[1][0], refPt[0][1]:refPt[1][1], :]
                    # goal_img = cv2.cvtColor(goal_img, cv2.COLOR_RGB2GRAY)
                    reason = ""  # input("Why did you click this point?\n")

                    # Debug >>>
                    

                    # Debug <<<
                    obs = worker.step(obs, goal_img, sim, agent, lRefPt, reason, save_path)
                    lRefPt = []
                if len(rRefPt) == 2:
                    worker.graph.addLandmark([rRefPt, worker.graph.current_node])
                    rRefPt = []

                    print('Landmarks:', worker.graph.landmarks)
                    if len(worker.graph.landmarks) > args.n_neighbors:
                        manifold_map = highManager.updateManifoldMap(worker.graph.getLandmarkGraph())
                        if args.display_map:
                            plt.clf()
                            plt.scatter(manifold_map[:, 0], manifold_map[:, 1])
                            plt.title("Current Manifold Map")
                            plt.pause(1)

            graph_path = save_path + 'worker_graph.json'
            worker.graph.save(graph_path); print('\n ' + graph_path + ' saved!')
            sim.close()
        else:
            print("Skipping", env_file)
