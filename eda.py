import numpy as np
from config import (
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    MAX_LENGTH
)
from data import load_data
from tokenizer import load_tokenizer
from filtering import filter_dataset


def print_dataset_info(train, validation, test):
    print("\n")
    print("DATASET INFO")

    print(f"Train:{len(train)}")
    print(f"Validation:{len(validation)}")
    print(f"Test:{len(test)}")


def calculate_lengths(dataset, tokenizer):
    source_lengths = []
    target_lengths = []

    for sample in dataset:
        source_text = sample["translation"][SOURCE_LANGUAGE]
        target_text = sample["translation"][TARGET_LANGUAGE]

        source_ids = tokenizer(
            source_text,
            add_special_tokens=False
        )["input_ids"]

        target_ids = tokenizer(
            target_text,    
            add_special_tokens=False
        )["input_ids"]

        source_lengths.append(len(source_ids))
        target_lengths.append(len(target_ids))

    return source_lengths, target_lengths


def print_statistics(name, lengths):
    print("\n")
    print(name)

    print(f"Mean:{np.mean(lengths):.2f}")
    print(f"Median:{np.median(lengths):.2f}")
    print(f"Minimum:{min(lengths)}")
    print(f"Maximum:{max(lengths)}")
    print(f"95 percentile:{np.percentile(lengths, 95):.2f}")


def print_truncation_statistics(source_lengths, target_lengths):
    source_truncated = np.sum(np.array(source_lengths) > MAX_LENGTH)
    target_truncated = np.sum(np.array(target_lengths) > MAX_LENGTH)

    source_percent = source_truncated / len(source_lengths) * 100
    target_percent = target_truncated / len(target_lengths) * 100

    print("\n")
    print(f"Sentences longer than {MAX_LENGTH} tokens")

    print(f"Source:{source_truncated}({source_percent:.2f}%)")
    print(f"Target:{target_truncated}({target_percent:.2f}%)")


def show_examples(dataset, n_examples=5):
    print("\n")
    print("SAMPLE TRANSLATIONS")

    for i in range(n_examples):
        sample = dataset[i]["translation"]

        print(f"\nExample {i + 1}")

        print(f"{SOURCE_LANGUAGE}:")
        print(sample[SOURCE_LANGUAGE])

        print()

        print(f"{TARGET_LANGUAGE}:")
        print(sample[TARGET_LANGUAGE])

        print("-" * 50)


def main():

    train, validation, test = load_data()

    train = filter_dataset(train)
    validation = filter_dataset(validation)
    test = filter_dataset(test)

    tokenizer = load_tokenizer()

    print_dataset_info(train, validation, test)

    source_lengths, target_lengths = calculate_lengths(
        train,
        tokenizer
    )

    print_statistics("Source language", source_lengths)

    print_statistics("Target language", target_lengths)

    print_truncation_statistics(
        source_lengths,
        target_lengths
    )

    show_examples(train)


if __name__ == "__main__":
    main()