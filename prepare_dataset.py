from datasets import load_dataset, concatenate_datasets
from filtering import filter_dataset
import os
from config import (
    DATASET_NAME,
    LANGUAGE_PAIR,
    SEED,
    USE_SUBSET,
    TRAIN_SAMPLES,
    VALIDATION_SAMPLES,
    TEST_SAMPLES,
    DATASET_PATH)


os.makedirs(DATASET_PATH, exist_ok=True)
dataset = load_dataset(DATASET_NAME, LANGUAGE_PAIR, streaming=True)


if USE_SUBSET:
    validation = filter_dataset(
        dataset["validation"],
        max_samples=VALIDATION_SAMPLES
    )

    test = filter_dataset(
        dataset["test"],
        max_samples=TEST_SAMPLES
    )

    validation_extra = VALIDATION_SAMPLES - len(validation)
    test_extra = TEST_SAMPLES - len(test)

    train_needed = (
        TRAIN_SAMPLES
        + validation_extra
        + test_extra
    )

    print(f"\nNeed from train: {train_needed}")
    print(f"Train: {TRAIN_SAMPLES}")
    print(f"Extra validation: {validation_extra}")
    print(f"Extra test: {test_extra}")

    train_pool = filter_dataset(dataset["train"].shuffle(seed=SEED), max_samples=train_needed)

    train = train_pool.select(
        range(0, TRAIN_SAMPLES)
    )

    if validation_extra > 0 or test_extra > 0:
        validation_extra_data = train_pool.select(
            range(TRAIN_SAMPLES, TRAIN_SAMPLES + validation_extra)
        )

        test_extra_data = train_pool.select(
            range(TRAIN_SAMPLES + validation_extra, train_needed)
        )

    if validation_extra > 0:
        validation = concatenate_datasets([
            validation,
            validation_extra_data
        ])

    if test_extra > 0:
        test = concatenate_datasets([
            test,
            test_extra_data
        ])

    validation = validation.shuffle(seed=SEED)
    test = test.shuffle(seed=SEED)

else:
    train = filter_dataset(dataset["train"])
    validation = filter_dataset(dataset["validation"])
    test = filter_dataset(dataset["test"])

print("\nFinal dataset sizes:")
print(f"Train: {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Test: {len(test)}")

train.save_to_disk(f"{DATASET_PATH}/train")

validation.save_to_disk(f"{DATASET_PATH}/validation")

test.save_to_disk(f"{DATASET_PATH}/test")