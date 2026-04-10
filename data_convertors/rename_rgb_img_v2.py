import os
import glob
from PIL import Image

####################
# Edit
dataset_v_ORI = 'LAVN'
dataset_v_DST = 'LAVN_dfv2'
'''
dfv2: each subfolder has only one image.
'''
datset_root = '/media/brcao/eData2/Data/datasets'
root = datset_root + '/' + dataset_v_ORI
dataset_ls = ['Gibson', 'Matterport', '480p_LMa']
name_len = 5
dest_root = datset_root + '/' + dataset_v_DST # for self-supervised learning
####################

for dataset in sorted(dataset_ls):
    dataset_path = f'{root}/{dataset}'
    for traj_path in sorted(glob.glob(dataset_path + '/*')):
        print('\n traj_path: ', traj_path)
        
        for img_path in glob.glob(traj_path + '/*.png'):
            if 'rgb' in img_path:
                print('\n img_path: ', img_path)
                file_name_ls = img_path.split('/')
                print('\nfile_name_ls: ', file_name_ls)
                # e.g. file_name_ls:  ['/media/brcao/eData2/Data/datasets/LAVN/Gibson/traj', 'Okabena/depth', 'v2', '343.png']
                img_name = file_name_ls[-1] # e.g. '343.png'
                img = Image.open(img_path)
                # img.show()
                img_name_str = str(img_name)
                while len(img_name_str) < 5 + 4:
                    img_name_str = '0' + img_name_str
                new_img_name = img_name_str.replace('.png', '.jpg')
                print('\nnew_img_name: ', new_img_name)
                new_img_folder_path = '/'.join(file_name_ls[:-1]).replace(f'{dataset_v_ORI}', f'{dataset_v_DST}')
                new_img_folder_path = new_img_folder_path[:new_img_folder_path.rindex('/')] + '_' + file_name_ls[-2] + '_' + new_img_name[:-4]
                print('\nnew_img_folder_path ', new_img_folder_path)
                if not os.path.exists(new_img_folder_path):
                    os.makedirs(new_img_folder_path); print(f'\n{new_img_folder_path} created!')

                new_img_file_path = new_img_folder_path + '/' + new_img_name
                print('\nnew_img_file_path: ', new_img_file_path)
                img.save(new_img_file_path); print(f'\n{new_img_file_path} saved!')

        # Same as previous one
        for img_path in glob.glob(traj_path + '/*.jpg'):
            if 'rgb' in img_path:
                print('\n img_path: ', img_path)
                file_name_ls = img_path.split('/')
                print('\nfile_name_ls: ', file_name_ls)
                # e.g. file_name_ls:  ['/media/brcao/eData2/Data/datasets/LAVN/Gibson/traj', 'Okabena/depth', 'v2', '343.png']
                img_name = file_name_ls[-1] # e.g. '343.png'
                img = Image.open(img_path)
                # img.show()
                img_name_str = str(img_name)
                while len(img_name_str) < 5 + 4:
                    img_name_str = '0' + img_name_str
                new_img_name = img_name_str.replace('.png', '.jpg')
                print('\nnew_img_name: ', new_img_name)
                new_img_folder_path = '/'.join(file_name_ls[:-1]).replace(f'{dataset_v_ORI}', f'{dataset_v_DST}')
                new_img_folder_path = new_img_folder_path[:new_img_folder_path.rindex('/')] + '_' + file_name_ls[-2] + '_' + new_img_name[:-4]
                print('\nnew_img_folder_path ', new_img_folder_path)
                if not os.path.exists(new_img_folder_path):
                    os.makedirs(new_img_folder_path); print(f'\n{new_img_folder_path} created!')

                new_img_file_path = new_img_folder_path + '/' + new_img_name
                print('\nnew_img_file_path: ', new_img_file_path)
                img.save(new_img_file_path); print(f'\n{new_img_file_path} saved!')
                