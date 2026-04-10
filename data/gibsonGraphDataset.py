import cv2
import numpy

from torch.utils.data import Dataset
from glob import glob

class LatentSpaceDataset(Dataset):
    """
    Args:
        :param data_path: folder closest to data/dataset/split/ where the graphs are stored
    """
    def __init__(self, data_path):
        self.data_path = data_path
        # assuming the data is organized data/dataset/split/environment/difficulty/contour/traj#/*.png
        self.files = glob(data_path+'**/.png', recursive=True) #should be all the image files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        #TODO: do we need to return the ground truth location as well???
        #TODO: need to return positives (if a batch of this is gotten randomly we'll have lots of negatives already)
        return cv2.cvtColor(cv2.imread(self.files[item]), cv2.COLOR_RGB2BGR)

if __name__=='__main__':
    #TODO: test dataset functionality
    print()