import evaluate
import numpy as np
from config import (
    BLEU_ENABLED,
    CHRF_ENABLED,
    COMET_ENABLED,
    COMET_MODEL
)
from comet import download_model, load_from_checkpoint

bleu_metric = None
chrf_metric = None
comet_model = None

def load_metrics():
    global bleu_metric, chrf_metric

    if BLEU_ENABLED and bleu_metric is None:
        bleu_metric = evaluate.load("sacrebleu")

    if CHRF_ENABLED and chrf_metric is None:
        chrf_metric = evaluate.load("chrf")

def load_comet():
    global comet_model

    if COMET_ENABLED and comet_model is None:
        print("Downloading COMET model...")

        model_path = download_model(
            "Unbabel/wmt22-comet-da"
        )

        comet_model = load_from_checkpoint(
            model_path
        )

    return comet_model


def decode_predictions(eval_preds, tokenizer):
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.where(
        predictions != -100,
        predictions,
        tokenizer.pad_token_id
    )

    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id
    )

    decoded_predictions = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True
    )

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )

    return decoded_predictions, decoded_labels

def calculate_metrics(predictions, references, sources=None):
    load_metrics()
    results = {}

    if BLEU_ENABLED:
        bleu = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
        results["BLEU"] = bleu["score"]

    if CHRF_ENABLED:
        chrf = chrf_metric.compute(predictions=predictions, references=references)
        results["chrF"] = chrf["score"]

    return results

def compute_metrics(eval_preds, tokenizer):
    predictions, references = decode_predictions(
            eval_preds,
            tokenizer
    )
    results = calculate_metrics(
        predictions,
        references
    )

    print("\nValidation metrics")

    for metric, value in results.items():
        print(f"{metric}: {value:.2f}")

    return results

def calculate_comet(predictions, references, sources):
    if not COMET_ENABLED:
        return None

    model = load_comet()

    """if comet_model is None:
        print("\nLoading COMET model...")
        comet_model = load_from_checkpoint(COMET_MODEL)"""

    data = [
        {
            "src": source,
            "mt": prediction,
            "ref": reference
        }
        for source, prediction, reference
        in zip(sources, predictions, references)
    ]

    output = comet_model.predict(
        data,
        batch_size=8,
        gpus=1
    )

    return output.system_score