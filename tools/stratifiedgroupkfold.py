import os
import pandas
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


root_dir = "/home/hal/Shijia/Struggle-aware-deep-models/EPIC-Struggle-Dataset/"

epic_pipe_dir = os.path.join(root_dir, "EPIC_Tower")
annotation_dir = os.path.join(root_dir, "annotation")
splits_dir = os.path.join(root_dir, 'splits', 'EPIC_Tower')
epic_pipe_annotation = os.path.join(annotation_dir, "UoB_str_tower.csv")


if __name__ == '__main__':

    vid_name_ls = os.listdir(epic_pipe_dir)
    print('Total number of videos', len(vid_name_ls))
    print('training set', int(round(len(vid_name_ls) * 3 / 4, 0)), 'test set', len(vid_name_ls)-int(round(len(vid_name_ls) * 3 / 4, 0)))
    df = pandas.read_csv(epic_pipe_annotation)

    participant_id_ls = [vid_name.split('_')[0] for vid_name in vid_name_ls]
    participant_id_ls = list(set(participant_id_ls))
    print('number of participants', len(participant_id_ls))
    participant_stat_dict = {}
    for participant_id in participant_id_ls:
        participant_stat_dict[participant_id] = {'count': 0, 'bin_count': [0, 0], 'four-way_count': [0, 0, 0, 0]}
        for vid_name in vid_name_ls:
            vid_id = vid_name.split('.')[0]
            row_index = df.index[df['VideoID'] == vid_id].tolist()[0]
            vid_label = df.loc[row_index, 'GA'] # 1, 2, 3, 4
            
            if participant_id == vid_name.split('_')[0]:
                participant_stat_dict[participant_id]['count'] += 1
                participant_stat_dict[participant_id]['four-way_count'][vid_label-1] += 1
                participant_stat_dict[participant_id]['bin_count'][0 if vid_label in [1, 2] else 1] += 1
            
    for k, v in participant_stat_dict.items():
        print(k, v['count'], v['bin_count'], v['four-way_count'])
    
    # Prepared for the StratifiedGroupKFold 
    X = []
    Y = []
    Groups = []
    for vid_id in vid_name_ls:
        vid_id = vid_id.split('.')[0]
        row_index = df.index[df['VideoID'] == vid_id].tolist()[0]
        vid_label = df.loc[row_index, 'GA'] # 1, 2, 3, 4
        vid_group = vid_id.split('_')[0]
        X.append(vid_id)
        Y.append(vid_label)
        Groups.append(vid_group)

    sgkf = StratifiedGroupKFold(n_splits=4)
    split_count = 0
    for train, test in sgkf.split(X, Y, groups=Groups):
        split_count += 1
        print('# training split', len(train), '# test split', len(test))
        # Training split statistics
        training_participants_id = []
        for i in train:
            training_participants_id.append(X[i].split('_')[0])
        training_participants_id_ls = list(set(training_participants_id))
        print(training_participants_id_ls)
        for training_pid in training_participants_id_ls:
            print(training_pid,
                  participant_stat_dict[training_pid]['count'],
                  participant_stat_dict[training_pid]['bin_count'],
                  participant_stat_dict[training_pid]['four-way_count'])
        # Test split statistics
        test_participants_id = []
        for i in test:
            test_participants_id.append(X[i].split('_')[0])
        test_participants_id_ls = list(set(test_participants_id))
        print(test_participants_id_ls)
        bin_count_ls = []
        four_way_count_ls = []
        for test_pid in test_participants_id_ls:
            print(test_pid,
                  participant_stat_dict[test_pid]['count'],
                  participant_stat_dict[test_pid]['bin_count'],
                  participant_stat_dict[test_pid]['four-way_count'])
            bin_count_ls.append(participant_stat_dict[test_pid]['bin_count'])
            four_way_count_ls.append(participant_stat_dict[test_pid]['four-way_count'])
        bin_count_arr = np.array(bin_count_ls)
        four_way_count_arr = np.array(four_way_count_ls)
        print('bin cls stats', np.sum(bin_count_arr, axis=0))
        print('four-way cls stats', np.sum(four_way_count_arr, axis=0))

        # Save the video IDs into .txt files
        with open(os.path.join(splits_dir, 'train_{}.txt'.format(int(split_count))), 'a') as f:
            for item in [X[i] for i in train]:
                f.write(item + '\n')
        with open(os.path.join(splits_dir, 'test_{}.txt'.format(int(split_count))), 'a') as f:
            for item in [X[i] for i in test]:
                f.write(item + '\n')
        # import pdb; pdb.set_trace()

