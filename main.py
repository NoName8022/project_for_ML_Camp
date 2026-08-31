from data import load_data
from tokenizer import load_tokenizer, tokenize_dataset
from model import create_model
from train import train_model, create_trainer
from evaluation import evaluate_model, show_examples
from config import SOURCE_LANGUAGE_MBART, TARGET_LANGUAGE_MBART

train, val, test = load_data()

raw_test = test

tokenizer = load_tokenizer()

token_train = tokenize_dataset(train, tokenizer)
token_val = tokenize_dataset(val, tokenizer)

model = create_model(tokenizer)

trainer = create_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=token_train,
    validation_dataset=token_val
)

trainer = train_model(trainer)

print("\nFINAL VALIDATION")
validation_results = trainer.evaluate()

print("\nFinal validation metrics:")
for metric, score in validation_results.items():
    if isinstance(score, (int, float)):
        print(f"{metric}: {score:.5f}")

print("\nTEST EVALUATION")

print("\nTEST SAMPLE")
for i in range(5):
    print(raw_test[i]["translation"])

results, predictions, references, sources = evaluate_model(
        trainer.model,
        tokenizer,
        raw_test
    )

print("src_lang:", SOURCE_LANGUAGE_MBART)
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

trainer.save_model("final_model")
tokenizer.save_pretrained("final_model")
