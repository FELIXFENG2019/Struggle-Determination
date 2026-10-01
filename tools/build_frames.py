"""Extract every frame of each 10-second clip into extracted_frames/<dataset>/<VideoID>/img_XXX.jpg.

Usage:
    python tools/build_frames.py --dataset_path /path/to/Struggle-Dataset
"""
import argparse
import os

import cv2
from tqdm import tqdm

from common import DATASETS, DEFAULT_ROOT


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_ROOT,
                        help='root of the downloaded Struggle-Dataset (contains Pipes-Struggle/, ...)')
    parser.add_argument('--out_path', type=str, default=None,
                        help='output folder (default: <dataset_path>/extracted_frames)')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS))
    return parser.parse_args()


def extract_frames(root_path, out_path, dataset_name):
    video_dir = os.path.join(root_path, dataset_name)
    if not os.path.isdir(video_dir):
        print('Skipping {}: {} not found'.format(dataset_name, video_dir))
        return
    video_list = sorted(v for v in os.listdir(video_dir) if v.lower().endswith('.mp4'))
    out_path = os.path.join(out_path, dataset_name)
    for vid in tqdm(video_list, desc=dataset_name):
        vid_name = os.path.splitext(vid)[0]
        out_path_sub = os.path.join(out_path, vid_name)
        if os.path.isdir(out_path_sub):
            continue
        os.makedirs(out_path_sub)
        video = cv2.VideoCapture(os.path.join(video_dir, vid))
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(count):
            ret, frame = video.read()
            assert ret, 'failed to read frame {} of {}'.format(i, vid)
            cv2.imwrite(os.path.join(out_path_sub, 'img_{:03d}.jpg'.format(i)), frame)
        video.release()


def main():
    args = get_args()
    out_path = args.out_path or os.path.join(args.dataset_path, 'extracted_frames')
    for dataset_name in args.datasets:
        print('Preparing {}'.format(dataset_name))
        extract_frames(args.dataset_path, out_path, dataset_name)
    print('Finished!')


if __name__ == '__main__':
    main()
