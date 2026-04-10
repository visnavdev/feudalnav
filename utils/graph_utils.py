import json
import os
from glob import glob

import habitat_sim
import numpy as np
from habitat_sim.errors import GreedyFollowerError
from habitat_sim.nav import GreedyGeodesicFollower

from graph import Graph
from habitat_utils import saveOBS

DIFFICULTY_THRESHOLDS = {None: [-np.inf, np.inf],
                         'easy': [1.5, 3],
                         'medium': [3, 5],
                         'hard': [5, 10]}


def computerForwardAction(start_loc, end_loc, base_movement=0.25):
    distance = ((end_loc[2] - start_loc[2]) ** 2 + (end_loc[0] - start_loc[0]) ** 2) ** 0.5
    return ['move_forward'] * int(np.ceil(distance // base_movement))
    # TODO: check to see if this is close enough? I think just has to be within 1m to count so this is actually too much?


def computeTurnAction(start_loc, end_loc, base_angle=15):
    # breakpoint()
    ang_diff0 = np.arctan2((end_loc[2] - start_loc[2]), (end_loc[0] - start_loc[0])) * 180 / np.pi
    # if ang_diff0 < 0:
    #     ang_diff360 = ang_diff0 + 360
    # else:
    #     ang_diff360 = ang_diff0 - 360
    if abs(ang_diff0) >= base_angle:  # and abs(ang_diff360) >= base_angle:
        times = ang_diff0 // base_angle
        print(times, ang_diff0, np.arctan2((end_loc[0] - start_loc[0]), (end_loc[2] - start_loc[2])) * 180 / np.pi)
        if ang_diff0 < 0:  # ang_diff360:
            # times = ang_diff0 // base_angle
            return ['turn_left'] * int(abs(times))
        else:
            # times = ang_diff360 // base_angle
            return ['turn_right'] * int(abs(times))
    return []


def generateTrajectoryGraph(folder_path, sim, sim_settings, difficulty=None, contour='straight', save=False):
    # set up variables and get initial path
    locations_path = folder_path
    diff_thresh = DIFFICULTY_THRESHOLDS[difficulty]
    path_data = getShortestPath(sim, folder_path)
    counter = 0
    distance_ratio = np.sum((path_data['start'] - path_data['end']) ** 2) ** 0.5 / path_data['distance']
    # TODO: check for rotational difference being <45 deg
    if contour == 'straight':
        # true if path meets curved criteria
        distance_check = distance_ratio >= 1.2
    else:
        # true if path meets straight criteria
        distance_check = distance_ratio < 1.2

    # if path not found or if path length not within difficulty thresholds, find a new one
    while path_data['path'] is None or path_data['distance'] < diff_thresh[0] or path_data['distance'] > diff_thresh[
        1] or distance_check:
        path_data = getShortestPath(sim, folder_path)
        counter += 1
        if counter > 100:
            print('Error: Generate Trajectory Graph timed out with path generation')
            return None

    # initializing the agent in the environment
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = path_data['start']
    agent.set_state(agent_state)
    agent_state = agent.get_state()
    agent_location = agent_state.position
    agent_orient = agent_state.rotation

    # getting the semantic actions necessary to follow the computed path
    try:
        follower = GreedyGeodesicFollower(sim.pathfinder, agent, goal_radius=1.99 * agent.agent_config.action_space[
            'move_forward'].actuation.amount)
        follower.find_path(path_data['end'])
        trajGraph = Graph(env_name=sim.curr_scene_name, goal_location=path_data['end'])
        # actions = orientTowardsFirstPoint(path_data['path'][0], path_data['path'][1])
        actions = follower.find_path(path_data['end'])  # getActionsForPath(path_data['path'], agent_orient)
    except GreedyFollowerError:
        print('Error: Something went wrong with getting the actions to follow the path to create the graph')
        return None  # to signify the graph couldn't be created

    assert (actions is not None)
    assert (actions != [])

    # make the folders for each trajectory predicted
    existing_traj_folders = glob(folder_path + '*')
    most_recent_traj = len(existing_traj_folders)
    folder_path += 'traj' + str(most_recent_traj) + '/'
    os.mkdir(folder_path)

    # follow the computed actions, make the graph, and save the observations
    # TODO: stop action not working! maybe fix???
    start_obs = sim.get_sensor_observations()
    if save:
        img_names = saveOBS(start_obs, folder_path, 0)
    else:
        img_names = ['', '']
    trajGraph.addNode(img_names[0], agent_location, agent_orient, img_names[1])

    end_act_ind = len(actions)
    for i, act in enumerate(actions):
        if act is not None:
            obs = sim.step(act)
            agent_state = agent.get_state()
            agent_loc = agent_state.position
            agent_orient = agent_state.rotation
            # cv2.imshow('', obs['color_sensor'])
            # cv2.waitKey(400)

            if save:
                img_names = saveOBS(obs, folder_path, i + 1)
            else:
                img_names = ['', '']
            trajGraph.addNode(img_names[0], agent_loc, agent_orient, img_names[1])
            if ((agent_loc[0] - path_data['end'][0]) ** 2 + (agent_loc[2] - path_data['end'][2]) ** 2) ** 0.5 <= \
                    agent.agent_config.action_space['move_forward'].actuation.amount:
                end_act_ind = i
                break

    trajGraph.actions = actions[:end_act_ind]

    # TODO: assuming goal image is last one collected from the node, but could need rotations once get there
    # should make a case where that's true sometimes and add more images/actions to the graph
    # ie. just change the rgb/depth images associated with the last node/goal node

    if save:
        with open(locations_path + 'location_track.json', 'r') as file:
            tracked_locations = json.load(file)
        tracked_locations['locations'].append((path_data['start'].tolist(), path_data['end'].tolist()))
        with open(locations_path + 'location_track.json', 'w') as file:
            json.dump(tracked_locations, file)
        trajGraph.save(folder_path + 'trajGraph.json')

    return trajGraph  # to signify that the graph was created successfully


def getActionsForPath(path, agent_orient=None):
    actions = []
    if agent_orient:
        agent_orient
        actions.extend(computeTurnAction(path[0], path[0]))
    for p in range(len(path) - 1):
        actions.extend(computeTurnAction(path[p], path[p + 1]))
        actions.extend(computerForwardAction(path[p], path[p + 1]))
    return actions


def getShortestPath(sim, folder_path, start=None, end=None):
    new = False
    if os.path.isfile(folder_path + "location_track.json"):
        with open(folder_path + 'location_track.json', 'r') as file:
            tracked_locations = json.load(file)
    else:
        tracked_locations = {'locations': []}
        with open(folder_path + 'location_track.json', 'w') as file:
            json.dump(tracked_locations, file)

    while not new:
        if start is None:
            start = sim.pathfinder.get_random_navigable_point()
            start[1] = sim.config.agents[0].height
        if end is None:
            end = sim.pathfinder.get_random_navigable_point()
            end[1] = sim.config.agents[0].height
        if (start.tolist(), end.tolist()) not in tracked_locations['locations']:
            new = True

    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = end
    found_path = sim.pathfinder.find_path(path)
    if found_path:
        return {'path': np.stack(path.points),
                'path_object': path,
                'distance': path.geodesic_distance,
                'start': start,
                'end': end}
    else:
        return {'path': None,
                'path_object': path,
                'distance': np.inf,
                'start': start,
                'end': end}


def orientTowardsFirstPoint(start, point):
    breakpoint()
    # eye = start
    # look = point
    up = np.array([0, 1, 0]).T
    zhat = start - point
