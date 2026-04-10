import argparse
import os
from glob import glob

import habitat_sim

from utils.graph_utils import generateTrajectoryGraph
from utils.habitat_utils import get_test_split_files, make_cfg


def getArgs():
    parser = argparse.ArgumentParser()

    # Superglue arguments
    parser.add_argument('--x_size', type=int, default=640, help='x dimension of the image')
    parser.add_argument('--y_size', type=int, default=480, help='y dimension of the image')

    # Habitat arguments
    parser.add_argument('--habitat_path', type=str, default='/home/faith/GitRepos/habitat/',
                        help='path to the folder with the habitat data')
    parser.add_argument('--data_path', type=str, default='/home/faith/Desktop/Red_Disk_Data/habitat_graph_data/',
                        help='path to the folder with the dataset graph data')
    parser.add_argument('--base_path', type=str, default='/home/faith/GitRepos/habitat_heuristic/',
                        help='base path to this folder')

    # Habitat run arguments
    parser.add_argument('--runs_per_level', type=int, default=25,
                        help='number of trajectories to make per level per contour')
    parser.add_argument('--save', action='store_true', help='whether or not to save the graphs')
    parser.add_argument('--difficulty', type=str, default='all', help='difficulty of the random paths')
    parser.add_argument('--dataset', type=str, default='gibson', choices=['gibson', 'matterport'],
                        help='which dataset to use')
    parser.add_argument('--dataset_split', type=list, default=['train'], choices=[['train', 'test'], ['test']],
                        help='difficulty of the random paths')

    return parser.parse_args()


if __name__ == '__main__':
    # get arguments
    args = getArgs()
    breakpoint()
    # make the folder for the chosen dataset and the data folder
    folder_path = args.data_path
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    folder_path = folder_path + args.dataset + '/'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # get list of all the environment files available from the current dataset
    if args.dataset == 'matterport':
        print('matterport data not downloaded on Faiths machine; use gibson instead')
        breakpoint()
    all_env_files = glob(args.habitat_path + args.dataset + '/*')

    # iterate over the chosen splits (train/test)
    for split in args.dataset_split:
        # get all file names belonging to the dataset's test scenes
        test_scene_files = get_test_split_files(env=args.dataset)

        # make the folder corresponding to the split
        split_path = folder_path + split + '/'
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        # get the environments belonging to the current split
        if split == 'train':
            split_env_files = [x for x in all_env_files if x.split('/')[-1] not in test_scene_files['all']]
        else:
            print('Warning! This doesnt have the functionality you want! You need to use the specific envs related to \
            the straight/curved split as well!')
            breakpoint()
            split_env_files = test_scene_files['all']

        split_env_files.sort()
        for env_num, env_scene_file in enumerate(split_env_files):
            print("\t\t\t\t\t\t\t\t\t Working on Environment", env_num + 1, "of", len(split_env_files))
            # configure the habitat environment
            try:
                sim_settings = {"scene_dataset_config_file": "default",
                                "scene": env_scene_file,
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

                # make the folder for the specific environment
                env_data_save_path = split_path + sim.curr_scene_name + '/'
                if not os.path.exists(env_data_save_path):
                    os.mkdir(env_data_save_path)

                # parse through the difficulty levels
                if args.difficulty == 'all':
                    difficulties = ['easy', 'medium', 'hard']
                else:
                    difficulties = [args.difficulty]

                # make the folders for the difficulty levels
                for level in difficulties:
                    level_path = env_data_save_path + level + '/'
                    if not os.path.exists(level_path):
                        os.mkdir(level_path)

                    # make a folder for the straight/curved split
                    for contour in ['straight', 'curved']:
                        print("\t\t\t\t\t\t\t Working on", sim.curr_scene_name, " level:", level, " contour:", contour,
                              "\n")
                        contour_path = level_path + contour + '/'
                        if not os.path.exists(contour_path):
                            os.mkdir(contour_path)

                        none_counter = 0
                        traj_folders = glob(contour_path + '*')
                        # make several training graphs for each environment for each difficulty level
                        while len(traj_folders) <= args.runs_per_level and none_counter < 100:
                            try:
                                trajGraph = generateTrajectoryGraph(contour_path, sim, sim_settings, difficulty=level,
                                                                    contour=contour, save=args.save)
                                if trajGraph is None:
                                    none_counter += 1
                                #     print(
                                #         "Error: traj graph not made for scene:" + sim.curr_scene_name + ", level: " + level + ", i: " + str(
                                #             i))

                                traj_folders = glob(contour_path + '*')
                            except:
                                none_counter += 1
                                traj_folders = glob(contour_path + '*')
                sim.close()
            except Exception as e:
                print(e)
                print('Moving on to next environment!!!')
                sim.close()
