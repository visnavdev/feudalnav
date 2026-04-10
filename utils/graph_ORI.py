import json

import cv2
import numpy as np


class Graph():
    def __init__(self, env_name, goal_location=None):
        self.nodes = []
        self.edges = []
        self.actions = []
        self.current_node = 0
        self.goal_location = goal_location
        self.landmarks = []
        self.env_name = env_name

    def addLandmark(self, data):
        # data is of the form [[x,y], node_number] where [x,y] is the image coord of the landmark and
        # node_num is the node whose observation contains the landmark
        self.landmarks.append(data)

    def addNode(self, img_path, location, orientation, depth_path=None, click_point=None, reason=None):
        node = Node(img_path, location, orientation, depth_path, click_point, reason)
        self.nodes.append(node)
        self.current_node = len(self.nodes) - 1
        self.updateEdges()

    def computeLandmarkDistance(self, node1Index, node2Index):
        if node1Index > self.current_node or node2Index > self.current_node or node1Index < 0 or node2Index < 0:
            return None
        else:
            # assuming traveled "shortest" geodesic distance between 2 consecutive landmark nodes so returning
            # the # of actions taken between the 2 nodes (which = the number of nodes between the landmark  nodes
            # because storing one node per action)
            # TODO: if more complicated landmark representation/edge connections than sequential nodes, will need to change this
            return abs(self.landmarks[node2Index][-1] - self.landmarks[node1Index][-1])

    def getLandmarkGraph(self):
        new_graph = Graph(env_name=self.env_name)
        for landmark in self.landmarks:
            node = self.nodes[landmark[-1] - 1]
            # where the click point is the point the person clicked in the image to indicate the landmark
            new_graph.addNode(node.img_path, node.location, node.orientation, node.depth_path, landmark[0])
            new_graph.landmarks = self.landmarks
        return new_graph

    def load(self, file_path):
        with open(file_path, 'r') as file:
            file_data = json.load(file)

        max_node_num = 0
        for key, val in file_data.items():
            if 'node' in key:
                num = int(key.split('node')[-1])
                if num > max_node_num:
                    max_node_num = num
            # elif 'edges' in key:
            #     self.edges = val
            # TODO: if edges ever get features, need to change this so that read them in directly instead of
            #  recomputing every time in addNode function
            elif 'goal_location' in key:
                self.goal_location = val
            elif 'landmarks' in key:
                self.landmarks = val
            elif 'action' in key:
                self.actions = val
            elif 'env_name' in key:
                self.env_name = val
        for i in range(max_node_num + 1):
            data = file_data['node' + str(i)]
            self.addNode(data['img_path'], data['location'], data['orientation'], data['depth_path'],
                         data['click_point'], data['reason'])

    def save(self, save_path):
        data = {}
        for i, node in enumerate(self.nodes):
            data['node' + str(i)] = node.getSaveDict()
        data['edges'] = self.edges
        if self.goal_location is not None:
            data['goal_location'] = self.goal_location.tolist()
        else:
            data['goal_location'] = None
        data['start_location'] = self.nodes[0].location.tolist()
        data['landmarks'] = self.landmarks
        data['actions'] = self.actions
        data['env_name'] = self.env_name
        with open(save_path, 'w') as f:
            json.dump(data, f)

    def updateEdges(self):
        if len(self.edges) == 0:
            self.edges.append([0])
        elif len(self.edges[0]) != len(self.nodes):
            for i in range(len(self.edges)):
                self.edges[i].append(0)
            self.edges.append([0] * len(self.nodes))
            if len(self.nodes) > 1:
                self.edges[-1][-2] = 1
                self.edges[-2][-1] = 1
        # TODO: is there some smarter way to make these graphs more connected?


class Node():
    def __init__(self, img_path, location, orientation=None, depth_path=None, click_point=None, reason=None):
        self.img_path = img_path
        self.depth_path = depth_path
        self.location = location
        # the orientation of the robot when the image was taken
        self.orientation = orientation
        self.click_point = np.array(click_point)
        self.reason = reason

    def getImage(self):
        if self.img_path:
            return cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        else:
            return None

    def getDepth(self):
        if self.depth_path:
            return cv2.imread(self.depth_path)  # TODO: might have to cast this as grayscale??? bc single value?
        else:
            return None

    def getSaveDict(self):
        return {'img_path': self.img_path,
                'depth_path': self.depth_path,
                'location': self.location.tolist(),
                'orientation': self.orientation.angle(),
                'click_point': self.click_point.tolist(),
                'reason': self.reason}
