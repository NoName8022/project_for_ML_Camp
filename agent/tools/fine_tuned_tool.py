import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from pathlib import Path
from langchain_core.tools import tool

SOURCE_LANGUAGE_MBART = "en_XX"
TARGET_LANGUAGE_MBART = "uk_UA"

BASE_MODEL = "facebook/mbart-large-50"

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "checkpoint-31250"


tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH)
)


base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL
)

model = PeftModel.from_pretrained(
    base_model,
    str(MODEL_PATH)
)


device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()


def translate(text: str) -> str:

    print("[MBART] translate() called")

    tokenizer.src_lang = SOURCE_LANGUAGE_MBART

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    print("[MBART] token count:", inputs["input_ids"].shape[1])

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            forced_bos_token_id=tokenizer.lang_code_to_id[
                TARGET_LANGUAGE_MBART
            ]
        )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    print("[MBART] RESULT:", repr(result))

    return result


@tool
def fine_tuned_model(text: str) -> str:
    """
    Translate English text into Ukrainian using
    the fine-tuned machine translation model.
    """
    print("\nFINE-TUNED MODEL")
    print("INPUT LENGTH:", len(text))
    print("INPUT:", text[:500])

    result = translate(text)

    print("OUTPUT:", result)

    return result

if __name__ == "__main__":
    print(
        translate("Hello, how are you?")
    )


