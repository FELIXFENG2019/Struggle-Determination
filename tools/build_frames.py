import os
import glob
import sys
from tqdm import tqdm


root_path = r"/home/shijia/Documents/Struggle-aware-deep-models/EPIC-Struggle-Dataset"


def extract_frames(root_path, out_path, dataset_name):
    print("Extracting Frames")
    epic_pipes_path = os.path.join(root_path, dataset_name)
    epic_pipes_vidlist = os.listdir(epic_pipes_path)
    out_path = os.path.join(out_path, dataset_name)
    with tqdm(total=len(epic_pipes_vidlist)) as pbar:
        for vid in epic_pipes_vidlist:
            vid_name = vid.split('.')[0]
            out_path_sub = os.path.join(out_path, vid_name)
            if not os.path.isdir(out_path_sub):
                # print("creating folder: " + out_path_sub)
                os.mkdir(out_path_sub)
                # extract frames first
                import cv2
                video = cv2.VideoCapture(os.path.join(epic_pipes_path, vid))
                count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in range(count):
                    ret, frame = video.read()
                    assert ret
                    cv2.imwrite('{}/img_{:03d}.jpg'.format(out_path_sub, i), frame)
            pbar.update()
    print("Finished!")


def main(root_path):
    out_path = os.path.join(root_path, 'extracted_frames')
    # prepare for the EPIC-Pipes dataset
    print("Preparing EPIC-Pipes Dataset")
    extract_frames(root_path, out_path, 'EPIC_Pipes')

    # prepare for the EPIC-Tent dataset
    print("Preparing EPIC-Tent Dataset")
    extract_frames(root_path, out_path, 'EPIC_Tent')

    # prepare for the EPIC-Tower dataset
    print("Preparing EPIC-Tower Dataset")
    extract_frames(root_path, out_path, 'EPIC_Tower')


if __name__ == '__main__':
    main(root_path)


