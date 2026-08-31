import torch
from tqdm import tqdm
from config import (
    NUM_BEAMS,
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    MAX_LENGTH,
    SOURCE_LANGUAGE_MBART,
    TARGET_LANGUAGE_MBART
)
from metrics import calculate_metrics, calculate_comet


def generate_predictions(model, tokenizer, test_dataset):
    model.eval()

    predictions = []
    references = []
    sources = []

    device = model.device
    tokenizer.src_lang = SOURCE_LANGUAGE_MBART
    forced_bos_token_id = tokenizer.lang_code_to_id[TARGET_LANGUAGE_MBART] 

    with torch.no_grad():
        for sample in tqdm(test_dataset, desc="Generating translations"):
            source = sample["translation"][SOURCE_LANGUAGE]
            #prompt = (f"translate English to Ukrainian: {source}")
            reference = sample["translation"][TARGET_LANGUAGE]

            inputs = tokenizer(
                #prompt,
                source,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH)

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()}

            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                decoder_start_token_id=forced_bos_token_id,
                num_beams=NUM_BEAMS,
                repetition_penalty=1.0,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                early_stopping=True,
                max_new_tokens=MAX_LENGTH)

            prediction = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True)

            predictions.append(prediction)
            references.append(reference)
            sources.append(source)

    return predictions, references, sources


def evaluate_model(model, tokenizer, test_dataset):
    predictions, references, sources = generate_predictions(
        model,
        tokenizer,
        test_dataset
    )

    results = calculate_metrics(
        predictions,
        references,
        sources
    )

    comet_score = calculate_comet(
        predictions,
        references,
        sources
    )

    if comet_score is not None:
        results["COMET"] = comet_score
    
    return results, predictions, references, sources


def show_examples(predictions, references, sources, num_examples=5):
    print("\n")
    print("Translation examples")

    num_examples = min(
        num_examples,
        len(predictions))

    for i in range(num_examples):
        print(f"\nExample {i + 1}")
        print(f"\nSource:\n{sources[i]}")
        print(f"\nReference:\n{references[i]}")
        print(f"\nPrediction:\n{predictions[i]}")
        print("\n")