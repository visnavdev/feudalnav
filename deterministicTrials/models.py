import argparse
import shutil
import sys

sys.path.append('../habitat_heuristic')

import cv2
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import Isomap
from matplotlib import pyplot as plt

# from SuperGluePretrainedNetwork.models.matching import Matching
# from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from utils.graph import Graph
from utils.habitat_utils import saveOBS


class Worker(object):
    def __init__(self, args, device, env_name):
        self.superglue_config = {
            'superpoint': {
                'nms_radius': args.nms_radius,
                'keypoint_threshold': args.keypoint_threshold,
                'max_keypoints': args.max_keypoints
            },
            'superglue': {
                'weights': args.superglue,
                'sinkhorn_iterations': args.sinkhorn_iterations,
                'match_threshold': args.match_threshold,
            }
        }
        self.device = device
        self.depth_threshold = args.depth_threshold
        # self.superglue = Matching(self.superglue_config).eval().to(device)
        self.x_size = args.x_size
        self.y_size = args.y_size
        self.x_threshold = args.x_size // 10
        self.y_threshold = args.y_size // 10
        self.graph = Graph(env_name=env_name)

    @torch.no_grad()
    # def identifyGoalPatch(self, obs, goal):
    #     # superglue code borrowed from here: https://github.com/magicleap/SuperGluePretrainedNetwork
    #     # get superglue to find salient points in obs and goal and match them
    #     # TODO: may need to resize the image to 480x640?
    #     if obs.shape[-1] > 1:
    #         obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    #     data = {'image0': torch.from_numpy(obs / 255.).float()[None, None].to(self.device),
    #             'image1': torch.from_numpy(goal / 255.).float()[None, None].to(self.device)}
    #     pred = self.superglue(data)
    #     pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    #     kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    #     matches, conf = pred['matches0'], pred['matching_scores0']
    #     valid = [i for i in range(len(matches)) if matches[i] > -1 and conf[i] >= 0.75]
    #     mkptsOBS = kpts0[valid]
    #     # mkptsGOAL = kpts1[matches[valid]]
    #     # mconf = conf[valid]
    #
    #     # find center of goal keypoint matches and return
    #     goal_center = np.round(np.mean(mkptsOBS, axis=0)).astype(int)
    #     return goal_center

    def moveTowardGoal(self, goal_center):
        moves = []
        # check if should turn left or right or not
        if abs(goal_center[0] - self.x_size // 2) > self.x_threshold:
            if goal_center[0] - self.x_size // 2 < 0:
                moves.append('turn_left')
            else:
                moves.append('turn_right')

        # TODO: check this logic!!! stop condition could be weird. what if item is on ground? should really checkf or depth instead.
        # maybe pass in obs???

        # check if should move forward or stop
        if goal_center[1] - self.y_size // 2 > self.y_threshold:
            pass
        else:
            moves.append('move_forward')

        return moves

    def step(self, obs, goal, sim=None, agent=None, click_point=None, reason=None, save_path=None):
        if save_path is None:
            save_path = ''
        # if len(goal.shape) > 2:
        #     goal_center = self.identifyGoalPatch(obs['color_sensor'], goal)
        # else:
        goal_center = np.array(goal)
        moves = self.moveTowardGoal(goal_center)

        if sim is None:
            return moves
        else:
            new_obs = obs
            print('Moves taken:', moves)
            for i, m in enumerate(moves):
                # breakpoint()
                # perform_discrete_collision_detection
                if not ("forward" in m and np.mean(
                        new_obs['depth_sensor'][int(self.y_size * 0.3):,
                        int(self.x_size * 0.3):int(self.x_size * 0.6)]) <= self.depth_threshold):
                    new_obs = sim.step(m)
                    img_names = saveOBS(new_obs, save_path, sim._num_total_frames)
                    agent_state = agent.get_state()
                    agent_location = agent_state.position
                    agent_orient = agent_state.rotation
                    self.graph.addNode(img_names[0], agent_location, agent_orient, img_names[1], click_point, reason)
                    self.graph.actions.append(m)
            return new_obs

    def reset(self, path):
        shutil.rmtree(path)


class OLDMidManager(object):
    def __init__(self, args, device):
        self.superpoint_config = {
            'nms_radius': args.nms_radius,
            'keypoint_threshold': args.keypoint_threshold,
            'max_keypoints': args.max_keypoints
        }
        self.superpoint = SuperPoint(self.superpoint_config).to(device)
        self.device = device
        self.distance_threshold = args.cluster_dist_thresh

    @torch.no_grad()
    def getPotentialGoal(self, obs, graph=None):
        # superpoint code borrowed from here: https://github.com/magicleap/SuperGluePretrainedNetwork
        if obs.shape[-1] > 1:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        data = {'image0': torch.from_numpy(obs / 255.).float()[None, None].to(self.device)}
        pred = self.superpoint({'image': data['image0']})
        kpts = pred['keypoints'][0]
        scores = pred['scores'][0]
        # descripts = pred['descriptors'][0]

        # only take the ones with scores above the mean; scores tell how "point-ish" a keypoint is
        avg = torch.mean(scores)
        valid = scores > avg
        mkpts = kpts[valid]
        mkpts = mkpts.cpu()

        # TODO: check to see if matching features between real goal and current obs
        # and if so, pick subgoal from features that overlap

        # TODO: if no goal features, can A) pick largest cluster of points and go there? B) pick furthest away set of 
        # points and go there? not quite sure actually

        if graph is None:  # or change this to if goal is not found in graph?
            # find portion of image with highest density of dots (for now)
            # TODO: arbitrarily picked 20 to best fit the test case; maybe should change that???
            clustering = AgglomerativeClustering(None, linkage='single', distance_threshold=self.distance_threshold)
            clusters = clustering.fit_predict(mkpts)
            largest_cluster = max(set(clusters), key=list(clusters).count)
            print('largest cluster:', largest_cluster)
            # plt.imshow(obs)
            # for i in range(max(clusters)):
            #     inds=[clusters==i]
            #     print(sum(sum(inds)))
            #     plt.scatter(mkpts[inds[0],0], mkpts[inds[0],1])
            # plt.savefig('gibson_superglue_test.png')
            # plt.show()
            points = mkpts[clusters == largest_cluster]
            min_coords = torch.min(points, axis=0)[0].int()
            max_coords = torch.max(points, axis=0)[0].int()
            goal_img = obs[min_coords[1]:max_coords[1], min_coords[0]:max_coords[0]]
            # plt.imshow(goal_img)
            # plt.show()
            return goal_img
        else:
            return None
            # TODO: figure out how to incorporate graph knowledge of goal location, robot location, and image
            # learned from the videos???

    def step(self, obs):
        goalimg = self.getPotentialGoal(obs)
        return goalimg


class HighManager(object):
    def __init__(self, args):
        self.graph = None
        self.current_map = None
        self.isomap = Isomap(n_neighbors=args.n_neighbors, metric="precomputed")

    def updateManifoldMap(self, graph):
        distance_matrix = np.zeros((len(graph.nodes), len(graph.nodes)))
        for i in range(graph.current_node + 1):
            for j in range(graph.current_node + 1):
                distance_matrix[i][j] = graph.computeLandmarkDistance(i, j)
            # distance_matrix[i+1][i] = graph.computeLandmarkDistance(i, i + 1)
        print(distance_matrix)
        manifold_map = self.isomap.fit_transform(distance_matrix)
        # TODO: how do we leverage the data from the current map to update this new one?
        # put that here
        self.current_map = manifold_map
        return manifold_map


# this is only here for the tests at the bottom of this file
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


if __name__ == '__main__':
    args = getArgs()
    # device = torch.device('cpu')
    # testimg = cv2.cvtColor(cv2.imread('test.png'), cv2.COLOR_BGR2GRAY)
    # goalimg = cv2.cvtColor(cv2.imread('goal.png'), cv2.COLOR_BGR2GRAY)
    #
    # # test upper level manager
    #
    # # test mid-level manager
    # midMan = MidManager(args, device)
    # goalimg = midMan.step(testimg)
    # plt.imshow(goalimg)
    # plt.show()
    #
    # # test worker
    # worker = Worker(args, device)
    # moves = worker.step(testimg, goalimg)
    # print(moves)

    # test high level manager
    graph = Graph(env_name=None)
    graph.load('test_imgs/worker_graph.json')
    landmark_graph = graph.getLandmarkGraph()

    highManager = HighManager(args)
    map_landmarks = highManager.updateManifoldMap(landmark_graph)
    print(map_landmarks)
    plt.figure()
    plt.scatter(map_landmarks[:, 0], map_landmarks[:, 1])
    plt.title("Landmark Map")
    plt.show()
