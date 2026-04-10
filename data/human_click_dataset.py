import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from itertools import product
import torch

class HumanClickData(Dataset):
    def __init__(self, data_folder_path):
        self.data_folder = data_folder_path
        self.file_paths = glob(data_folder_path+'*/rgb*.png')
        self.crop_boundaries = [3,4]

    def __len__(self):
        return len(self.file_paths)

    def centeredDistanceMatrix(self, n, m, point):
        # make sure n is odd
        x, y = np.meshgrid(range(n), range(m))
        return np.sqrt((x - point[0] + 1) ** 2 + (y - point[1] + 1) ** 2)

    def __getitem__(self, item):
        # read images
        image = cv2.imread(self.file_paths[item])

        # get graph data to get clicked point
        with open(self.file_paths[item].split('rgb')[0]+'worker_graph.json') as f:
            graph_data = json.load(f)

        node_info = graph_data['node'+str(int(self.file_paths[item].split('rgb_')[-1].split('.')[0])-2)]

        # create probability map for the image with the click point = 0 and radially decreasing
        point = node_info['click_point']
        percentages = self.centeredDistanceMatrix(image.shape[0], image.shape[1], point)
        percentages = 1 - (percentages - np.min(percentages)) / (np.max(percentages) - np.min(percentages))

        # crop both the image and the percentages

        # gives back [3, 4, 3, 160, 160]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_crops = torch.tensor(image).unfold(0, image.shape[0] // self.crop_boundaries[0],
                image.shape[0] // self.crop_boundaries[0]).unfold(1,image.shape[1] // self.crop_boundaries[1],
                image.shape[1] // self.crop_boundaries[1])
        prob_crops = torch.tensor(percentages).unfold(0, image.shape[0] // self.crop_boundaries[0],
                image.shape[0] // self.crop_boundaries[0]).unfold(1,image.shape[1] // self.crop_boundaries[1],
                image.shape[1] // self.crop_boundaries[1])
        # now changing to shape [-1, 3, 160, 160]
        image_crops=image_crops.reshape(-1,3,image.shape[0] // self.crop_boundaries[0],image.shape[1] // self.crop_boundaries[1])
        prob_crops=prob_crops.reshape(-1,image.shape[0] // self.crop_boundaries[0],image.shape[1] // self.crop_boundaries[1])
        mean_crop_probabilities = torch.softmax(torch.mean(prob_crops,axis=2).mean(axis=1),-1)

        return image_crops, mean_crop_probabilities

if __name__=='__main__':
    dataset = HumanClickData('/Users/faith_johnson/GitRepos/habitat_heuristic/test_imgs/')
    dataset.__getitem__(1)
