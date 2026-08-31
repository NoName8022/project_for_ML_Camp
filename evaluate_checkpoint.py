from data import load_data
from tokenizer import load_tokenizer, tokenize_dataset
from evaluation import evaluate_model, show_examples

from transformers import (
    MBartForConditionalGeneration,
    Seq2SeqTrainer
)

from train import create_training_arguments

from config import (
    SOURCE_LANGUAGE_MBART,
    TARGET_LANGUAGE_MBART
)


CHECKPOINT_PATH = "checkpoints/checkpoint-31250"

train, val, test = load_data()

tokenizer = load_tokenizer()

model = MBartForConditionalGeneration.from_pretrained(
    CHECKPOINT_PATH
)

token_val = tokenize_dataset(
    val,
    tokenizer
)


training_args = create_training_arguments()


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=token_val,
    processing_class=tokenizer
)


print("\nFINAL VALIDATION")

validation_results = trainer.evaluate()

for metric, score in validation_results.items():
    if isinstance(score, (int, float)):
        print(f"{metric}: {score:.5f}")

print("\nTEST EVALUATION")

results, predictions, references, sources = evaluate_model(
    model,
    tokenizer,
    test
)


print("\nsrc_lang:", SOURCE_LANGUAGE_MBART)
print("target_lang:", TARGET_LANGUAGE_MBART)

print(
    "target id:",
    tokenizer.lang_code_to_id[TARGET_LANGUAGE_MBART]
)


print("\nTest results")

for metric, score in results.items():
    print(f"{metric}: {score:.5f}")


show_examples(
    predictions,
    references,
    sources,
    num_examples=10
)