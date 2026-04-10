import os
import glob

gibson_root = '/media/brcao/eData2/Data/datasets/LAVN/Gibson'

for path_ in glob.glob(gibson_root + '/*'):
    print('\npath_: ', path_)
    for path_2 in glob.glob(path_ + '/*'):
        print('\npath_2: ', path_2)
        cmd = f'mv {path_2} {gibson_root}'
        os.system(cmd)
        print(cmd)
    if '001' in path_:
        cmd = f'rm -r {path_}'
        os.system(cmd)
        print(cmd)