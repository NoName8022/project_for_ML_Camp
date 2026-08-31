from datasets import load_from_disk
from config import DATASET_PATH

def load_data():
    train = load_from_disk(f"{DATASET_PATH}/train")
    validation = load_from_disk(f"{DATASET_PATH}/validation")
    test = load_from_disk(f"{DATASET_PATH}/test")
    return train, validation, test