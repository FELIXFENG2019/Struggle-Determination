# Struggle Determination Datasets

Struggle determination is an essential part of the project, enabling the wearable assistive intelligent system to detect struggle actions during the egocentric view of hand-object interactions and offer appropriate instructional advice to help the user. We created three datasets in the tasks (both indoors and outdoors) including assembling plumbing pipes, tent pitching, and tower of Hanoi game, named respectively as Pipes-Struggle, Tent-Struggle, and Tower-Struggle. 

## Data Collection

We recruited different numbers of individual participants to perform each of the tasks following diagram of instructions during which the participants wore a head-mounted GoPro camera to record first-person videos involving scences of hand-object interactions. The videos were captured at $1920\times1080$ resolution, at either 30 or 60hz. The raw videos are then uniformly trimmed into 10-second segments, with resolution resized into $456\times256$. Struggle determination annotation was accomplished by having both MTurk workers and human experts to rate the trimmed video segments in different struggle levels. 

## Description of the data

In this section, we describe how our struggle determination dataset is organised and how the data is saved in each file. Here is an overview of the dataset structure:
```
.
└── Struggle-Dataset/
    ├── annotation/
    │   ├── pipe.csv
    │   ├── tent.csv
    │   ├── tent_subactions/
    │   │   ├── tent_0_ass_sup.csv
    │   │   ├── tent_1_ins_sta.csv
    │   │   ├── tent_2_ins_sup.csv
    │   │   ├── tent_3_ins_tab.csv
    │   │   └── tent_9_pla_guy.csv
    │   └── tower.csv
    ├── Pipes-Struggle/
    │   └── <Pipes_VideoIDs>.MP4 e.g. 01_00_0001.MP4
    ├── Tent-Struggle/
    │   └── <Tent_VideoIDs>.MP4 e.g. 08_02_00.MP4
    ├── Tower-Struggle/
    │   └── <Tower_VideoIDs>.MP4 e.g. 01_00_0000.MP4
    ├── extracted_frames/
    │   ├── Pipes-Struggle/
    │   │   └── <Pipes_VideoIDs>/
    │   │       ├── img_000.jpg
    │   │       ├── img_001.jpg
    │   │       └── ...
    │   ├── Tent-Struggle/
    │   │   └── <Tent_VideoIDs>/
    │   │       ├── img_000.jpg
    │   │       ├── img_001.jpg
    │   │       └── ...
    │   └── Tower-Struggle/
    │       └── <Tower_VideoIDs>/
    │           ├── img_000.jpg
    │           ├── img_001.jpg
    │           └── ...
    ├── splits/
    │   ├── Pipes-Struggle/
    │   │   ├── test_<num_splits>.txt
    │   │   └── train_<num_splits>.txt
    │   ├── Tent-Struggle/
    │   │   ├── test_<num_splits>.txt
    │   │   └── train_<num_splits>.txt
    │   └── Tower-Struggle/
    │       ├── test_<num_splits>.txt
    │       └── train_<num_splits>.txt
    ├── tools/
    │   ├── build_frames.py
    │   ├── stratifiedgroupkfold.py
    │   └── human_baseline_stats.py
    └── README.md
```

### Folder Structure
- Annotation:
  This folder contains annotations of the struggle determination dataset. There are three annotation files that correspond to the tasks plumbing pipes, pitching tent, and tower of Hanoi game by the names of pipes.csv, tent.csv, and tower.csv respectively. There is a folder called 'tent_subactions' which contains the annotations files by sub-actions of pitching tent task named as following:
  - tent_0_ass_sup.csv
  - tent_1_ins_sta.csv
  - tent_2_ins_sup.csv
  - tent_3_ins_sup.csv
  - tent_9_ins_sup.csv
  
  See more details of sub-actions for the tent pitching task in the 'Action-annotation' dictionary below.

  Description of each column in the file:
  - VideoID: This represents the video ID for each individual video in an activity (e.g., VideoID.MP4). VideoID is defined as ParticipantID_RecordID_10SecondClipID.
  - Vote#: This represents a crowd's vote collected by the Amazon Mechanical Turk service.
  There are 20 votes for the same video clip (except Tower-Struggle: 15). The scale of vote is from 1 to 4. (1: definitely non-struggle, 2: slightly non-struggle, 3: slightly struggle, 4: definitely struggle)
  - StdDev: This represents the standard deviation of the crowd's multiple votes.
  - Mode: This represents the mode statistics (the most frequently selected option) out of - crowd's multiple votes. 
  - GA: Golden Annotation (GA) is a single vote chosen by an expert on the same video.

- Tent-Struggle:
  This folder contains a set of 10-second video segments collected from tent pitching task
  (equivalent to the 'EPIC-Tent' dataset [^1]: https://github.com/youngkyoonjang/EPIC_Tent2019).
  The subfolders correspond to the Action_annotaion dictionary as follows:
  ```
  Action_annotation = {0:'assemble support', 1:'insert stake', 2:'insert support', 3:'insert support tab', 4:'instruction', 5:'pickup/open stakebag', 6:'pickup/open supportbag', 7:'pickup/open tentbag', 8:'pickup/place ventcover', 9:'place guyline', 10:'spread tent', 11:'tie top'} 
  ```
  The annotations of Tent-Struggle only contain actions 0, 1, 2, 3, 9 in the EPIC-Tent dataset [^1].
- Pipes-Struggle:
  This folder contains a set of 10-second video segments collected from plumbing pipes task.
- Tower-Struggle:
  This folder contains a set of 10-second video segments collected from tower of Hanoi task.
- Extracted Frames:
  This folder contains extracted frames in JPG format from the video samples from three struggle determination datasets: Pipes-Struggle, Tent-Struggle, and Tower-Struggle. 
- Splits:
  This folder contains the four-fold training and testing splits for cross-validation.
- Tools:
  - build_frames.py is used to extract frames from the video samples.
  - stratifiedgroupkfold.py is used to split the four-fold test splits in a [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) method.
  - human_baseline_stats.py is used to calculate the human baseline accuracy in each of the test splits.

## Download link

* **Google Drive** (International)
    * [Struggle-Dataset (.zip)](https://drive.google.com/file/d/1nVwLPNVcVsvvCJDlnyYYwulmezeEPgbY/view?usp=sharing)
    * **Size:** 28.6 GB
* **Baidu NetDisk / 百度网盘** (Mainland China)
    * [Struggle-Dataset.tar.gz](https://pan.baidu.com/s/1apgIudPZGAWqSwgKu1ashw?pwd=2d8k)
    * **Size:** 28.4 GB
* [**Hugging Face**](https://huggingface.co/datasets/Shijia2025/Struggle-Dataset)

## Contributors

* Shijia Feng
* Michael Wray
* Brian Sullivan
* Youngkyoon Jang
* Casimir Ludwig
* Iain Gilchrist
* Walterio Mayol-Cuevas (Corresponding Author)

## License

This project is licensed under the Non-Commercial Government Licence for public sector information, found [here](https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/).

## Citation to this work

```
@misc{feng2024strugglingdatasetbaselinesstruggle,
      title={Are you Struggling? Dataset and Baselines for Struggle Determination in Assembly Videos}, 
      author={Shijia Feng and Michael Wray and Brian Sullivan and Youngkyoon Jang and Casimir Ludwig and Iain Gilchrist and Walterio Mayol-Cuevas},
      year={2024},
      eprint={2402.11057},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2402.11057}, 
}

@article{feng2025you,
  title={Are you struggling? dataset and baselines for struggle determination in assembly videos},
  author={Feng, Shijia and Wray, Michael and Sullivan, Brian and Jang, Youngkyoon and Ludwig, Casimir and Gilchrist, Iain and Mayol-Cuevas, Walterio},
  journal={International Journal of Computer Vision},
  pages={1--38},
  year={2025},
  publisher={Springer}
}
```

[^1]: Y. Jang, B. Sullivan, C. Ludwig, I. D. Gilchrist, D. Damen, and W. Mayol-Cuevas. Epic-tent: An egocentric video dataset for camping tent assembly. In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pages 4461–4469, Los Alamitos, CA, USA, oct 2019. IEEE Computer Society.




