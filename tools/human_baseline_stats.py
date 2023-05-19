import os
import argparse
import numpy as np
import pandas


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str,
                        default='/home/alexa/Shijia/Struggle-aware-deep-models/EPIC-Struggle-Dataset/')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    for epic_task in ['EPIC_Pipes', 'EPIC_Tent', 'EPIC_Tower']:
        print('Dataset ' + epic_task)

        for split in [1, 2, 3, 4]:
            print('test split {}'.format(split))
            # test video list directories
            vid_split_dir = os.path.join(args.dataset_path, 'splits', epic_task, 'test_{}.txt'.format(split))
            # annotation directories for different tasks
            if epic_task == 'EPIC_Tent':
                annotation_dir = os.path.join(args.dataset_path, 'annotation', 'UoB_str_tent.csv')
                num_voters = 20
            elif epic_task == 'EPIC_Pipes':
                annotation_dir = os.path.join(args.dataset_path, 'annotation', 'UoB_str_pipe.csv')
                num_voters = 20
            elif epic_task == 'EPIC_Tower':
                annotation_dir = os.path.join(args.dataset_path, 'annotation', 'UoB_str_tower.csv')
                num_voters = 15
            
            with open(vid_split_dir) as file:
                vid_list = [line.rstrip() for line in file]
            
            for num_classes in [2, 4]:
                print('Number of classes: {}'.format(num_classes))
                acc_per_vid = 0
                for vid in vid_list:
                    if not os.path.exists(os.path.join(args.dataset_path, 'extracted_frames', epic_task, vid)):
                        continue
                    df = pandas.read_csv(annotation_dir)

                    row_index = df.index[df['VideoID'] == vid].tolist()[0]
                    vid_label = df.loc[row_index, 'GA'] # Using Golden Annotation 1, 2, 3, and 4  

                    if num_classes == 4:
                        # manipulate golden annotations
                        ga_label = np.array([vid_label - 1])
                        # manipulate voters annotations
                        voters_labels = []
                        for i in range(num_voters):
                            voter_individual = df.loc[row_index, 'Vote{}'.format(i+1)]
                            voters_labels.append(voter_individual-1)
                    elif num_classes == 2:
                        # manipulate golden annotations
                        if vid_label == 1 or vid_label == 2:
                            # non-struggle
                            ga_label = np.array([0])
                        elif vid_label == 3 or vid_label == 4:
                            # struggle
                            ga_label = np.array([1])
                        # manipulate voters annotations
                        voters_labels = []
                        for i in range(num_voters):
                            voter_individual = df.loc[row_index, 'Vote{}'.format(i+1)]
                            if voter_individual == 1 or voter_individual == 2:
                                # non-struggle
                                voters_labels.append(0)
                            elif voter_individual == 3 or voter_individual == 4:
                                # struggle
                                voters_labels.append(1)
                    voters_labels_arr = np.array(voters_labels)
                    acc_per_vid += sum(voters_labels_arr == ga_label) / len(voters_labels)
                acc_total = acc_per_vid / len(vid_list) * 100
                print('Human Baseline Accuracy: {:.2f}%'.format(acc_total))
           
