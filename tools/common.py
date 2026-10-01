"""Shared constants and helpers for the Struggle Determination datasets."""
import os

import pandas

# Dataset folder name -> (annotation file, number of crowd voters per clip)
DATASETS = {
    'Pipes-Struggle': ('pipe.csv', 20),
    'Tent-Struggle': ('tent.csv', 20),
    'Tower-Struggle': ('tower.csv', 15),
}

NUM_SPLITS = 4

# Struggle levels used by both crowd votes and the Golden Annotation (GA)
LEVELS = {
    1: 'definitely non-struggle',
    2: 'slightly non-struggle',
    3: 'slightly struggle',
    4: 'definitely struggle',
}

# Default dataset root: the repository root (one level above tools/)
DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_binary(level):
    """Map a 4-level struggle score (1-4) to binary: 0 non-struggle, 1 struggle."""
    return 0 if level in (1, 2) else 1


def load_annotation(root, dataset):
    """Load the annotation CSV of a dataset, indexed by VideoID."""
    csv_name, _ = DATASETS[dataset]
    # VideoIDs such as '01_00_0001' must stay strings to keep leading zeros
    df = pandas.read_csv(os.path.join(root, 'annotation', csv_name), dtype={'VideoID': str})
    return df.set_index('VideoID')


def load_split(root, dataset, subset, split):
    """Return the list of VideoIDs in splits/<dataset>/<subset>_<split>.txt."""
    path = os.path.join(root, 'splits', dataset, '{}_{}.txt'.format(subset, split))
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]
