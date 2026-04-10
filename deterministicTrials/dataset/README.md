# LAVN Dataset
### Data Organization

The root folder ```LAVN_Dataset``` consists of three folders, including (1) ```src```, (2) ```Virtual``` and (3) ```Real```:
```
LAVN_Dataset
   |--src
      |--makeData_virtual.py
      |--makeData_real.py
   |--Virtual
      |--Gibson
         |--traj_<SCENE_ID>
            |--worker_graph.json
            |--rgb_<FRAME_ID>.jpg
            |--depth_<FRAME_ID>.jpg
         |--traj_Ackermanville
            |--worker_graph.json
            |--rgb_00001.jpg
            |--rgb_00002.jpg
            ...
            |--depth_00001.jpg
            |--depth_00002.jpg
            ...
         ...
      |--Matterport
         |--traj_<SCENE_ID>
            |--worker_graph.json
            |--rgb_<FRAME_ID>.jpg
            |--depth_<FRAME_ID>.jpg
         |--traj_00000-kfPV7w3FaU5
            |--worker_graph.json
            |--rgb_00001.jpg
            |--rgb_00002.jpg
            ...
            |--depth_00001.jpg
            |--depth_00002.jpg
            ...
         ...
   |--Real
      |--original
      |--480p
      |--480p_LMa
         |--worker_graph.json
         |--traj_480p_<SCENE_ID>
            |--rgb_<FRAME_ID>.jpg
         |--traj_480p_scene00
            |--rgb_00001.jpg
```
where the main landmark annotation scripts ```makeData_virtual.py``` and ```makeData_real.py``` are in folder (1) ```src```. (2) ```Virtual``` and (3) ```Real``` stores trajectories collecetd in the simulation and real world, respectively. In each trajectory's data is collected in the following format:
```
         |--traj_<SCENE_ID>
            |--worker_graph.json
            |--rgb_<FRAME_ID>.jpg
            |--depth_<FRAME_ID>.jpg
```
where ```<SCENE_ID>``` matches exactly the original one in [Gibson](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md) and [Matterport](https://aihabitat.org/datasets/hm3d/) run by the photo-realistic simulator [Habitat](https://github.com/facebookresearch/habitat-sim). Images are saved in either ```.jpg``` or ```.png``` format. Note that ```rgb``` images are the main visual representation while ```depth``` is the auxiliary visual information captured only in the virtual environment.

```worker_graph.json``` stores the meta data in dictionary in Python saved in ```json``` file with the following format:

```
{"node<NODE_ID>":
  {"img_path": "./human_click_dataset/traj_<SCENE_ID>/rgb_<FRAME_ID>.jpg",
   "depth_path": "./human_click_dataset/traj_<SCENE_ID>/depth_<FRAME_ID>.png",
   "location": [<LOC_X>, <LOC_Y>, <LOC_Z>],
   "orientation": <ORIENT>,
   "click_point": [<COOR_X>, <COOR_Y>],
   "reason": ""},
  ...
 "node0":
  {"img_path": "./human_click_dataset/traj_00101-n8AnEznQQpv/rgb_00002.jpg",
   "depth_path": "./human_click_dataset/traj_00101-n8AnEznQQpv/depth_00002.jpg",
   "location": [0.7419548034667969, -2.079209327697754, -0.5635206699371338],
   "orientation": 0.2617993967423121,
   "click_point": [270, 214],
   "reason": ""}
 ...
 "edges":...
 "goal_location": null,
 "start_location": [<LOC_X>, <LOC_Y>, <LOC_Z>],
 "landmarks": [[[<COOR_X>, <COOR_Y>], <FRAME_ID>], ...],
 "actions": ["ACTION_NAME", "turn_right", "move_forward", "turn_right", ...]
 "env_name": <SCENE_ID>
}
```
where ```[<LOC_X>, <LOC_Y>, <LOC_Z>]``` is the 3-axis location vector, ```<ORIENT>``` is the orientation only in simulation. ```[<COOR_X>, <COOR_Y>]``` are the image coordinates of landmarks. ```ACTION_NAME``` stores the action of the robot take from the current frame to the next frame.
