"""Regenerate the four-fold train/test splits with StratifiedGroupKFold.

Clips are stratified by their Golden Annotation (GA) and grouped by participant
(the first field of the VideoID), so no participant appears in both train and test.
Running this on the released annotation files reproduces splits/ exactly:
    python tools/stratifiedgroupkfold.py --out_dir /tmp/splits
"""
import argparse
import os
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold

from common import DATASETS, DEFAULT_ROOT, NUM_SPLITS, load_annotation, to_binary


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_ROOT,
                        help='folder containing annotation/ (default: repository root)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='where to write <dataset>/train_<k>.txt and test_<k>.txt')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS))
    return parser.parse_args()


def label_stats(labels):
    four_way = Counter(labels)
    binary = Counter(to_binary(l) for l in labels)
    return [binary[0], binary[1]], [four_way[l] for l in (1, 2, 3, 4)]


def make_splits(df, splits_dir):
    X = list(df.index)
    Y = list(df['GA'])
    groups = [vid_id.split('_')[0] for vid_id in X]
    print('Total number of videos', len(X), '| number of participants', len(set(groups)))

    os.makedirs(splits_dir, exist_ok=True)
    sgkf = StratifiedGroupKFold(n_splits=NUM_SPLITS)
    for split_count, (train, test) in enumerate(sgkf.split(X, Y, groups=groups), 1):
        test_participants = sorted({groups[i] for i in test})
        bin_count, four_way_count = label_stats([Y[i] for i in test])
        print('split {}: # training {}, # test {}, test participants {}'.format(
            split_count, len(train), len(test), test_participants))
        print('  test bin cls stats', bin_count, 'four-way cls stats', four_way_count)

        for subset, idx in (('train', train), ('test', test)):
            with open(os.path.join(splits_dir, '{}_{}.txt'.format(subset, split_count)), 'w') as f:
                for i in idx:
                    f.write(X[i] + '\n')


def main():
    args = get_args()
    for dataset in args.datasets:
        print('Dataset ' + dataset)
        make_splits(load_annotation(args.dataset_path, dataset), os.path.join(args.out_dir, dataset))


if __name__ == '__main__':
    main()
