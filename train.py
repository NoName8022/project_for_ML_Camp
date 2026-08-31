from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from functools import partial
from metrics import compute_metrics
from config import (
    NUM_BEAMS,
    MAX_LENGTH,
    OUTPUT_DIR,
    EVAL_STRATEGY,
    SAVE_STRATEGY,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_EPOCHS,
    LOGGING_STEPS,
    FP16,
    GRADIENT_CHECKPOINTING,
    WARMUP_STEPS,
    OPTIMIZER
)

def create_training_arguments():
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        fp16=FP16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={
        "use_reentrant": False},
        predict_with_generate=True,
        save_total_limit=4,
        load_best_model_at_end=False,
        logging_strategy="steps",
        #greater_is_better=True,
        #metric_for_best_model="chrF",
        generation_num_beams=NUM_BEAMS,
        generation_max_length=MAX_LENGTH,
        weight_decay=0.01,
        ddp_find_unused_parameters=False,
        #warmup_ratio=WARMUP_RATIO,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="linear",
        disable_tqdm=True,
        optim = OPTIMIZER
    )

    return training_args

def create_trainer(model, tokenizer, train_dataset, validation_dataset):
    training_args = create_training_arguments()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer)
    )

    return trainer
    

def train_model(trainer):
    trainer.train()
    return trainer