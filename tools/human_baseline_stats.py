"""Human baseline: agreement of individual crowd votes with the Golden Annotation (GA)
on each test split, for both the 2-class and 4-class settings.

Only the annotation/ and splits/ folders are needed, so this runs directly in a clone of the repo:
    python tools/human_baseline_stats.py
"""
import argparse
import os

import numpy as np

from common import DATASETS, DEFAULT_ROOT, NUM_SPLITS, load_annotation, load_split, to_binary


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_ROOT,
                        help='folder containing annotation/ and splits/ (default: repository root)')
    parser.add_argument('--require_frames', action='store_true',
                        help='only count clips whose extracted_frames/<dataset>/<VideoID> folder exists')
    return parser.parse_args()


def main():
    args = get_args()
    for dataset, (_, num_voters) in DATASETS.items():
        print('Dataset ' + dataset)
        df = load_annotation(args.dataset_path, dataset)
        vote_cols = ['Vote{}'.format(i + 1) for i in range(num_voters)]

        for split in range(1, NUM_SPLITS + 1):
            print('test split {}'.format(split))
            vid_list = load_split(args.dataset_path, dataset, 'test', split)
            if args.require_frames:
                vid_list = [vid for vid in vid_list if os.path.exists(
                    os.path.join(args.dataset_path, 'extracted_frames', dataset, vid))]

            for num_classes in [2, 4]:
                print('Number of classes: {}'.format(num_classes))
                acc_per_vid = 0
                for vid in vid_list:
                    ga_label = df.loc[vid, 'GA']  # Golden Annotation 1, 2, 3, 4
                    voters_labels = df.loc[vid, vote_cols].to_numpy()
                    if num_classes == 2:
                        ga_label = to_binary(ga_label)
                        voters_labels = np.array([to_binary(v) for v in voters_labels])
                    acc_per_vid += np.mean(voters_labels == ga_label)
                acc_total = acc_per_vid / len(vid_list) * 100
                print('Human Baseline Accuracy: {:.2f}%'.format(acc_total))


if __name__ == '__main__':
    main()
