'''
Usage:
python makeData_real.py --save_env_name --env_name 480p_scene01
'''

import argparse
import os
import random
from glob import glob

import cv2
from models import Worker, HighManager
from matplotlib import pyplot as plt
from utils.habitat_utils import make_cfg, saveRealOBS

def getArgs():
    parser = argparse.ArgumentParser()

    # Data collection arguments
    # parser.add_argument('--action_limit', type=int, default=500, help='max number of environment steps to make')

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

    # File arguments
    parser.add_argument('--data_path', type=str, default='/Data/datasets/LMNav/480p',
                        help='path to the folder with the dataset graph data') # edit
    parser.add_argument('-ds', '--dataset', type=str, default='real', choices=['real'], help='which dataset to use')
    parser.add_argument('-dst', '--dataset_type', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--env_name', type=str, default='None', help='env name')
    parser.add_argument('--depth_threshold', type=float, default=0.4,
                        help='how close the agent is allowed to get to objects in the sim in meters')
    parser.add_argument('--save_root', type=str, default='./human_click_dataset', help='Save data root')
    parser.add_argument('--absolute_scene_path', type=str, default='unspecified')
    parser.add_argument('-dm', '--display_map', action='store_true')
    parser.add_argument('-sen', '--save_env_name', action='store_true')
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
    print('\nlRefPt: ', lRefPt)
    print('\nrRefPt: ', rRefPt)


if __name__ == '__main__':
    device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = getArgs()

    if args.absolute_scene_path == 'unspecified':
        env_file_path = args.data_path + '/' + args.env_name
    else: env_file_path = args.absolute_scene_path
    print('\n env_file_path: ', env_file_path)

    ##################
    # Save folder >>>
    if not os.path.exists(args.save_root): os.mkdir(args.save_root)

    # make the folders for each trajectory predicted
    existing_traj_folders = glob(args.save_root + '/*')
    if args.save_env_name:
        save_path = args.save_root + '/traj_' + args.env_name
    else:
        most_recent_traj = str(len(existing_traj_folders))
        while len(most_recent_traj) < 5: most_recent_traj = '0' + most_recent_traj
        save_path = args.save_root + '/traj_' + most_recent_traj
        print('\n save_path: ', save_path)
    os.mkdir(save_path)
    # Save folder <<<
    ##################

    worker = Worker(args, device, args.env_name)
    highManager = HighManager(args)
    if not os.path.exists(args.save_root): os.mkdir(args.save_root)

    ############################
    # Loop over real images >>>
    image_paths = sorted(glob(env_file_path + '/*'))
    print('\nimage_paths: ', image_paths) # debug

    # Initialization
    cv2.namedWindow('Current Frame')
    cv2.setMouseCallback('Current Frame', click_and_crop)
    rRefPt = []
    lRefPt = []
    for frame_i, image_path in enumerate(image_paths):
        if frame_i + 1 > len(image_paths) - 1:
            # Last frame, no next frame
            break
        print('\nimage_path: ', image_path)
        img_curr = cv2.imread(image_path)
        img_next = cv2.imread(image_paths[frame_i + 1])
        
        action = None
        cv2.imshow('Current Frame', img_curr)
        cv2.imshow('Next Frame', img_next)

        ###########################################
        # Keyboards for saving actions as node >>>
        # Save node
        
        k = cv2.waitKey(0)
        if k == ord('q'):  # q key to stop
            break
        # elif k == 2424832:
        #     print('Left arrow.')
        # elif k == 2490368:
        #     print('Up arrow')
        # elif k == 2555904:
        #     print('Right arrow')
        # elif k == 2621440:
        #     print('Down arrow')
        elif k == ord('1'):
            action = 'turn_left'; print(action)
        elif k == ord('2'):
            action = 'move_forward'; print(action)
        elif k == ord('3'):
            action = 'turn_right'; print(action)

        frame_i_str = str(frame_i)
        while len(frame_i_str) < 5: frame_i_str = '0' + frame_i_str
        image_save_path = save_path + '/rgb_' + frame_i_str + '.jpg' # '.png'
        location = None
        orientation = None
        depth_path = None
        click_point = None
        reason = None
        worker.graph.addNode(image_save_path, location, orientation, depth_path, click_point, reason)
        worker.graph.actions.append(action)
        # Keyboards for saving actions as node <<<
        ###########################################

        # Save real image
        saveRealOBS(img_curr, save_path, frame_i)

        ###############################
        # Left click for landmarks >>>
        if len(lRefPt) > 0:
            worker.graph.addLandmark([lRefPt, worker.graph.current_node])
            lRefPt = []

            print('Landmarks:', worker.graph.landmarks)
            if len(worker.graph.landmarks) > args.n_neighbors:
                manifold_map = highManager.updateManifoldMap(worker.graph.getLandmarkGraph())
                if args.display_map:
                    plt.clf()
                    plt.scatter(manifold_map[:, 0], manifold_map[:, 1])
                    plt.title("Current Manifold Map")
                    plt.pause(1)
        # Left click for landmarks <<<
        ###############################
    graph_path = save_path + '/worker_graph.json'
    worker.graph.save(graph_path); print('\n ' + graph_path + ' saved!')











