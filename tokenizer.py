from transformers import AutoTokenizer

from config import (
    MODEL_NAME,
    MAX_LENGTH,
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    SOURCE_LANGUAGE_MBART,
    TARGET_LANGUAGE_MBART
)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    return tokenizer


def preprocess_function(examples, tokenizer):
    inputs = [
        #f"Translate English to Ukrainian: {sample[SOURCE_LANGUAGE]}"
        sample[SOURCE_LANGUAGE]
        for sample in examples["translation"]
    ]

    targets = [
        sample[TARGET_LANGUAGE]
        for sample in examples["translation"]
    ]

    tokenizer.src_lang = SOURCE_LANGUAGE_MBART
    tokenizer.tgt_lang = TARGET_LANGUAGE_MBART

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    labels["input_ids"] = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in sequence
        ]
        for sequence in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def tokenize_dataset(dataset, tokenizer):
    return dataset.map(
        lambda examples: preprocess_function(
            examples,
            tokenizer
        ),
        batched=True,
        remove_columns=dataset.column_names
    )