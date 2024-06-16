"""
Read our custom-collected json files
"""
import os
from glob import glob
import json
import re
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def delete_superfluous_files():
    path = f'/home/qiyuan/2023fall/Gait'
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.startswith('._.DS_Store') or name.startswith('._gait_') or name.startswith('.DS_Store'):
                file_path = os.path.join(root, name)
                os.remove(file_path)
                print(f'Deleted: {file_path}')


def read_json(seq_len=None):
    """
    read json file and save to pickle
    """
    path = f'/home/qiyuan/2023fall/Gait/gait_dataset_keypoints/output_json_folder'
    pose_folders = []
    # view_label = 0
    views = ['front', 'side']
    label_mapping = {'after_activity': 1,
                     'baseline': 0}
    data = []
    # labels = []

    pattern = f'/user_*/session_*/*/*/'
    matching_folders = glob(path + pattern, recursive=True)

    print("Matching folders:")
    for folder in matching_folders:
        if count_files(folder):
            match = re.search(r'user_(\d+)/session_(\d+)/(\w+)/gait_(\w+)', folder)
            if match:
                pose_folders.append(folder)
                user_id, session_id, activity_instance, view_label = match.groups()
                # print(folder, user_id, session_id, label_mapping[activity_instance], view_label, count_files(folder))  # split by matching_folders
                data.append((read_pose_from_json(folder, seq_len), user_id, session_id, activity_instance, view_label))  # seq_len is 300
                # labels.append((user_id, session_id, activity_instance, view_label))

    with open(f'output/fatigue_gait.pickle', 'wb') as f:
        pickle.dump(data, f)
    return data


def read_pose_from_json(path_to_json, num_sequence=None):
    # Load keypoint data from JSON output
    column_names = ['frames', 'label']
    # Paths - should be the folder where Open Pose JSON output was stored
    # Import Json files, pos_json = position JSON
    # json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    json_files = find_json_files(path_to_json)
    #need to sort the files so that loaded frames maintains sequence
    json_files.sort()
    #print('Found: ', len(json_files), 'json keypoint frame files')
    count = 0
    # instanciate dataframes
    frames = []
    data = []
    label='fatigued'
    # Loop through all json files in output directory
    for file in json_files:
        #print(file)
        #temp_df = json.load(open(path_to_json+file))
        with open(file) as f:
            # print(f)
            data = json.load(f)
            if len(data['people']) > 0:
                data = np.array(data['people'][0]['pose_keypoints_2d']) #.reshape(-1, 3)
                frames.append(data)
            else:
                continue
    #include the first 300 frames of sequences
    if num_sequence:
        frames = frames[:num_sequence]
    frames = np.array(frames, dtype=np.float32)  # .flatten()
    return frames


def find_json_files(folder):
    json_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                json_files.append(file_path)
    return json_files


def read_fatigue_gait():
    with open('output/fatigue_gait.pickle', 'rb') as f:
        loaded_data = pickle.load(f)
        print(loaded_data)


def count_files(directory):
    num_files = 0
    for root, dirs, files in os.walk(directory):
        num_files += len(files)
    return num_files


class FatigueGait(Dataset):
    def __init__(
        self,
        data_path,
        view='front',
        # train=True,
        transform=None,
        target_transform=None,
    ):
        """
        view: front or side, front min,max_len=305,618; side min,max_len=150,457
        sequence_length: no need
        """
        super(FatigueGait, self).__init__()
        with open(data_path, 'rb') as f:
            self.loaded_data = pickle.load(f)
        
        self.viewed_data = [x for x in self.loaded_data if x[-1] == f'{view}view']
        self.data = []
        self.labels = []
        self.label_mapping = {'after_activity': 1, 'baseline': 0, 'frontview': 2, 'sideview': 3}
        self.max_length = max([x[0].shape[0] for x in self.viewed_data])
        for i, seq in enumerate(self.viewed_data):
            num_pad_rows = self.max_length - seq[0].shape[0]
            pad_array = np.tile(seq[0][-1], (num_pad_rows, 1))
            padded_array = np.concatenate((seq[0], pad_array), axis=0).reshape(3,-1,25)
            self.data.append(padded_array)  # .reshape((1,3,-1))
            self.labels.append((int(seq[1]),int(seq[2]),self.label_mapping[seq[3]], 
                                self.label_mapping[seq[4]], seq[0].shape[0]))

        self.transform = transform
        self.target_transform = target_transform

        self.data_dict = {}
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.data[index]
        user_id, session_id, label, view, length = self.labels[index]
        return x, np.float32(label)


if __name__ == '__main__':
    # delete_superfluous_files()
    # read_json()
    # read_fatigue_gait()
    dataset = FatigueGait('output/fatigue_gait.pickle', view='side')
