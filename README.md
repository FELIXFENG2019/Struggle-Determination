# Struggle Determination Datasets

[![Paper (IJCV 2025)](https://img.shields.io/badge/Paper-IJCV%202025-blue)](https://link.springer.com/article/10.1007/s11263-025-02559-4)
[![arXiv](https://img.shields.io/badge/arXiv-2402.11057-b31b1b)](https://arxiv.org/abs/2402.11057)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/Shijia2025/Struggle-Dataset)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

**The struggle determination datasets proposed in the paper [Are you Struggling? Dataset and Baselines for Struggle Determination in Assembly Videos](https://link.springer.com/article/10.1007/s11263-025-02559-4) (International Journal of Computer Vision, 2025).**

Struggle determination enables a wearable assistive system to detect when a person is struggling during hand-object interactions, seen from an egocentric (first-person) view, and to offer instructional help at the right moment. We release three datasets covering indoor and outdoor assembly tasks — plumbing pipes, tent pitching and the Tower of Hanoi — named **Pipes-Struggle**, **Tent-Struggle** and **Tower-Struggle**.

**Contents:** [At a glance](#at-a-glance) · [Download](#download) · [Quick start](#quick-start) · [Data collection](#data-collection) · [Dataset structure](#dataset-structure) · [Annotation format](#annotation-format) · [Splits and evaluation](#splits-and-evaluation) · [Tools](#tools) · [License](#license) · [Citation](#citation)

## At a glance

| Dataset | Task | Clips (10 s) | Participants | Votes per clip | GA level 1 / 2 / 3 / 4 |
| --- | --- | ---: | ---: | ---: | --- |
| Pipes-Struggle | Assembling plumbing pipes | 1,011 | 23 | 20 | 146 / 230 / 400 / 235 |
| Tent-Struggle  | Pitching a tent (outdoors) | 585 | 21 | 20 | 110 / 154 / 209 / 112 |
| Tower-Struggle | Tower of Hanoi game | 236 | 20 | 15 | 28 / 19 / 52 / 137 |
| **Total** | | **1,832** | | | |

Struggle levels: **1** definitely non-struggle, **2** slightly non-struggle, **3** slightly struggle, **4** definitely struggle. GA = Golden Annotation by an expert.

Every clip comes with crowd votes from Amazon Mechanical Turk *and* an expert label, so the datasets support 2-class (struggle vs. non-struggle) and 4-class classification as well as regression or label-distribution learning.

## Download

The annotations, splits and tools are in this repository. The videos and extracted frames (~28 GB) can be downloaded from:

| Source | Link | Size |
| --- | --- | --- |
| Hugging Face | [Shijia2025/Struggle-Dataset](https://huggingface.co/datasets/Shijia2025/Struggle-Dataset) | |
| Google Drive (international) | [Struggle-Dataset.zip](https://drive.google.com/file/d/1nVwLPNVcVsvvCJDlnyYYwulmezeEPgbY/view?usp=sharing) | 28.6 GB |
| Baidu NetDisk / 百度网盘 (mainland China) | [Struggle-Dataset.tar.gz](https://pan.baidu.com/s/1apgIudPZGAWqSwgKu1ashw?pwd=2d8k) (code: `2d8k`) | 28.4 GB |

Download from Hugging Face on the command line:

```bash
pip install -U huggingface_hub
huggingface-cli download Shijia2025/Struggle-Dataset --repo-type dataset --local-dir Struggle-Dataset
```

## Quick start

The annotations and splits can be used straight from a clone of this repository, without downloading the videos:

```bash
git clone https://github.com/FELIXFENG2019/Struggle-Determination.git
cd Struggle-Determination
pip install -r requirements.txt
python tools/human_baseline_stats.py   # human baseline accuracy on every test split
```

Loading the labels of one train/test split in Python:

```python
import pandas as pd

root = "Struggle-Determination"            # or the root of the downloaded Struggle-Dataset
ann = pd.read_csv(f"{root}/annotation/pipe.csv", dtype={"VideoID": str}).set_index("VideoID")

def read_split(name):
    with open(f"{root}/splits/Pipes-Struggle/{name}.txt") as f:
        return [line.strip() for line in f if line.strip()]

train_ids, test_ids = read_split("train_1"), read_split("test_1")
y4 = ann.loc[train_ids, "GA"]              # 4-class labels: 1..4
y2 = (y4 >= 3).astype(int)                 # 2-class labels: 0 non-struggle, 1 struggle
# frames of a clip: <root>/extracted_frames/Pipes-Struggle/<VideoID>/img_000.jpg, img_001.jpg, ...
```

> Read `VideoID` as a string (`dtype={"VideoID": str}`), otherwise leading zeros such as `01_00_0001` may be lost.

## Data collection

Participants performed each task following a diagram of instructions while wearing a head-mounted GoPro camera, recording first-person videos of hand-object interactions. The videos were captured at $1920\times1080$ resolution at either 30 or 60 Hz. The raw videos were uniformly trimmed into 10-second segments and resized to $456\times256$. Struggle was annotated by both Amazon Mechanical Turk workers and human experts, who rated each segment on four struggle levels.

## Dataset structure

```
Struggle-Dataset/
├── annotation/
│   ├── pipe.csv
│   ├── tent.csv
│   ├── tower.csv
│   └── tent_subactions/
│       ├── tent_0_ass_sup.csv   # assemble support
│       ├── tent_1_ins_sta.csv   # insert stake
│       ├── tent_2_ins_sup.csv   # insert support
│       ├── tent_3_ins_tab.csv   # insert support tab
│       └── tent_9_pla_guy.csv   # place guyline
├── Pipes-Struggle/              # <VideoID>.MP4, e.g. 01_00_0001.MP4
├── Tent-Struggle/               # <VideoID>.MP4, e.g. 08_02_00.MP4
├── Tower-Struggle/              # <VideoID>.MP4, e.g. 01_00_0000.MP4
├── extracted_frames/
│   ├── Pipes-Struggle/<VideoID>/img_000.jpg, img_001.jpg, ...
│   ├── Tent-Struggle/<VideoID>/img_000.jpg, ...
│   └── Tower-Struggle/<VideoID>/img_000.jpg, ...
├── splits/
│   ├── Pipes-Struggle/{train,test}_{1,2,3,4}.txt
│   ├── Tent-Struggle/{train,test}_{1,2,3,4}.txt
│   └── Tower-Struggle/{train,test}_{1,2,3,4}.txt
├── tools/
│   ├── build_frames.py
│   ├── stratifiedgroupkfold.py
│   └── human_baseline_stats.py
└── README.md
```

The video folders and `extracted_frames/` are only in the download; everything else is also in this repository.

- **Pipes-Struggle / Tower-Struggle:** 10-second clips of the plumbing pipes and Tower of Hanoi tasks.
- **Tent-Struggle:** 10-second clips of the tent pitching task, taken from the [EPIC-Tent](https://github.com/youngkyoonjang/EPIC_Tent2019) dataset [^1]. EPIC-Tent's action labels are:
  ```
  {0: 'assemble support', 1: 'insert stake', 2: 'insert support', 3: 'insert support tab', 4: 'instruction',
   5: 'pickup/open stakebag', 6: 'pickup/open supportbag', 7: 'pickup/open tentbag', 8: 'pickup/place ventcover',
   9: 'place guyline', 10: 'spread tent', 11: 'tie top'}
  ```
  Tent-Struggle only contains actions 0, 1, 2, 3 and 9. `annotation/tent.csv` holds all of them; `annotation/tent_subactions/` holds the same rows split by action (49, 52, 204, 146 and 134 clips).
- **extracted_frames:** every frame of each clip as JPG, produced by `tools/build_frames.py`.
- **splits:** four-fold train/test splits for cross-validation (see below).

## Annotation format

Each annotation CSV has one row per clip:

| Column | Description |
| --- | --- |
| `VideoID` | `ParticipantID_RecordID_ClipID`; the clip is `<VideoID>.MP4`. |
| `Vote1` … `VoteN` | Struggle level (1–4) from each crowd worker on Amazon Mechanical Turk. N = 20 (Pipes, Tent) or 15 (Tower). |
| `StdDev` | Standard deviation of the crowd votes. |
| `Mode` | Most frequent crowd vote. |
| `GA` | Golden Annotation: a single struggle level (1–4) chosen by an expert. |

For 2-class experiments, levels 1–2 map to *non-struggle* and 3–4 to *struggle*.

## Splits and evaluation

`splits/<Dataset>/train_k.txt` and `test_k.txt` (k = 1…4) list the VideoIDs of four folds built with [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html): folds are stratified by `GA` and grouped by participant, so **no participant appears in both the training and test set of a fold**. Every clip appears in exactly one test fold.

To compare with the paper, train on `train_k`, evaluate on `test_k` with `GA` as the label, and report the accuracy averaged over the four folds.

## Tools

All tools take `--help`. `requirements.txt` lists their dependencies.

| Script | Purpose | Needs videos? |
| --- | --- | --- |
| `tools/build_frames.py --dataset_path <root>` | Extracts every frame of each clip to `extracted_frames/`. | Yes |
| `tools/stratifiedgroupkfold.py --out_dir <dir>` | Regenerates the four-fold splits from the annotations; the output matches `splits/`. | No |
| `tools/human_baseline_stats.py` | Human baseline: agreement of individual crowd votes with the GA on each test split, for 2 and 4 classes. | No |

## Contributors

* Shijia Feng
* Michael Wray
* Brian Sullivan
* Youngkyoon Jang
* Casimir Ludwig
* Iain Gilchrist
* Walterio Mayol-Cuevas (corresponding author)

Questions, problems with a download link, or found an error in the data? Please [open an issue](https://github.com/FELIXFENG2019/Struggle-Determination/issues).

## License

The datasets are released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license; see [LICENSE](LICENSE). They may be used for non-commercial research purposes with attribution.

## Citation

If you use these datasets, please cite:

```bibtex
@article{feng2025you,
  title={Are you struggling? dataset and baselines for struggle determination in assembly videos},
  author={Feng, Shijia and Wray, Michael and Sullivan, Brian and Jang, Youngkyoon and Ludwig, Casimir and Gilchrist, Iain and Mayol-Cuevas, Walterio},
  journal={International Journal of Computer Vision},
  pages={1--38},
  year={2025},
  publisher={Springer},
  doi={10.1007/s11263-025-02559-4}
}
```

<details>
<summary>arXiv preprint</summary>

```bibtex
@misc{feng2024strugglingdatasetbaselinesstruggle,
      title={Are you Struggling? Dataset and Baselines for Struggle Determination in Assembly Videos},
      author={Shijia Feng and Michael Wray and Brian Sullivan and Youngkyoon Jang and Casimir Ludwig and Iain Gilchrist and Walterio Mayol-Cuevas},
      year={2024},
      eprint={2402.11057},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2402.11057},
}
```
</details>

[^1]: Y. Jang, B. Sullivan, C. Ludwig, I. D. Gilchrist, D. Damen, and W. Mayol-Cuevas. EPIC-Tent: An egocentric video dataset for camping tent assembly. In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pages 4461–4469, 2019.
