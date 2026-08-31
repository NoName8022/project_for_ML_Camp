from transformers import MBartForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from config import (
    MODEL_NAME,
    MAX_LENGTH,
    NUM_BEAMS
)


def create_model(tokenizer):
    model = MBartForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj"
        ],
    )

    model = get_peft_model(
        model,
        lora_config
    )

    if hasattr(model, "_hf_tp_plan"):
        model._hf_tp_plan = None

    if hasattr(model, "_hf_device_mesh"):
        model._hf_device_mesh = None 

    model.generation_config.max_new_tokens = MAX_LENGTH
    model.generation_config.num_beams = NUM_BEAMS
    model.generation_config.no_repeat_ngram_size = 2

    model.print_trainable_parameters()

    return model