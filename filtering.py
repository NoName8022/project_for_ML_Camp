import os
import urllib.request
import fasttext

from datasets import Dataset

from config import (
    TARGET_LANGUAGE,
    LANG_CONFIDENCE_THRESHOLD,
)

LANG_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
LANG_MODEL_PATH = "lid.176.ftz"

lang_model = None


def get_language_model():
    global lang_model

    if lang_model is None:
        if not os.path.exists(LANG_MODEL_PATH):
            print("Downloading FastText model...")
            urllib.request.urlretrieve(
                LANG_MODEL_URL,
                LANG_MODEL_PATH,
            )

        lang_model = fasttext.load_model(LANG_MODEL_PATH)

    return lang_model


def is_target_language(text: str) -> bool:
    model = get_language_model()

    labels, confidences = model.predict(
        text.replace("\n", " ")
    )

    language = labels[0].replace("__label__", "")

    return (
        language == TARGET_LANGUAGE
        and confidences[0] >= LANG_CONFIDENCE_THRESHOLD
    )


def filter_dataset(dataset, max_samples=None):
    filtered_examples = []

    for example in dataset:
        text = example["translation"][TARGET_LANGUAGE]

        if not text or len(text.strip()) < 2:
            continue

        if is_target_language(text):
            filtered_examples.append(example)

        if (
            max_samples is not None
            and len(filtered_examples) >= max_samples
        ):
            break

    print(f"Collected {len(filtered_examples)} samples.")

    return Dataset.from_list(filtered_examples)