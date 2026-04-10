import habitat_sim
import numpy as np
from habitat_sim.nav import GreedyGeodesicFollower

from utils.graph import Graph
from utils.habitat_utils import saveOBS


class DeterministicWorker:
    def __init__(self, agent, folder_path=None, env_name=None):
        self.graph = Graph(env_name=env_name)
        self.agent = agent
        self.start_location = agent.get_state().position
        self.folder_path = folder_path

    def planPath(self, sim, point):
        point[1] = self.start_locationstart[1]
        path = habitat_sim.ShortestPath()
        path.requested_start = self.start_locationstart
        path.requested_end = point
        found_path = sim.pathfinder.find_path(path)
        if found_path:
            path_data = {'path': np.stack(path.points),
                         'path_object': path,
                         'distance': path.geodesic_distance,
                         'start': self.start_location,
                         'end': point}
        else:
            return None

    def navigateToPoint(self, sim, point):
        path_data = self.planPath(sim, point)
        if path_data is not None:
            follower = GreedyGeodesicFollower(sim.pathfinder, self.agent,
                                              goal_radius=1.99 * self.agent.agent_config.action_space[
                                                  'move_forward'].actuation.amount)
            actions = follower.find_path(path_data['end'])

            # TODO: what if actions is empty???

            start_obs = sim.get_sensor_observations()
            img_names = saveOBS(start_obs, self.folder_path, self.agent.get_state().position)
            self.graph.addNode(img_names[0], self.agent.get_state().position, self.agent.get_state().rotation,
                               img_names[1])

            for i, act in enumerate(actions):
                if act is not None:
                    obs = sim.step(act)
                    agent_state = self.agent.get_state()
                    agent_loc = agent_state.position
                    agent_orient = agent_state.rotation
                    if 'forward' in act:
                        img_names = saveOBS(obs, self.folder_path, agent_loc)
                        self.graph.addNode(img_names[0], agent_loc, agent_orient, img_names[1])
                    if ((agent_loc[0] - path_data['end'][0]) ** 2 + (agent_loc[2] - path_data['end'][2]) ** 2) ** 0.5 <= \
                            1 * self.agent.agent_config.action_space['move_forward'].actuation.amount:
                        end_act_ind = i
                        break

            actions = actions[:end_act_ind]
            return True
        else:
            return False

    def saveGraph(self):
        self.graph.save(self.folder_path + 'trajGraph.json')
