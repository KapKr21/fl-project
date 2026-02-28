from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import yaml
import numpy as np
import evaluate
import random
import torch

# Config

NUM_LAYERS = 12
MODEL_NAME = "roberta-base"
USE_LORA = False
SEEDS = [42, 123, 999]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
metric = evaluate.load("accuracy")


# Utilities
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


def layer_indices(layer_cfg):
    mode = layer_cfg["mode"]

    if mode == "explicit":
        return layer_cfg["indices"]

    elif mode == "top_k":
        k = layer_cfg["top_k"]
        return list(range(NUM_LAYERS - k, NUM_LAYERS))

    elif mode == "range":
        start, end = layer_cfg["range"]
        return list(range(start, end + 1))

    elif mode == "all":
        return list(range(NUM_LAYERS))

    else:
        raise ValueError(f"Unknown layer selection mode: {mode}")


def build_target_modules(selected_layers, target_projections):
    modules = []
    for layer_id in selected_layers:
        for proj in target_projections:
            modules.append(
                f"roberta.encoder.layer.{layer_id}.attention.self.{proj}"
            )
    return modules


def tokenize_mnli(dataset):
    def tokenize(batch):
        tokenized = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        tokenized["labels"] = batch["label"]
        return tokenized

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )
    dataset.set_format("torch")
    return dataset


# Data
def load_mnli():
    mnli = load_dataset("glue", "mnli")
    train = tokenize_mnli(mnli["train"])
    val_matched = tokenize_mnli(mnli["validation_matched"])
    val_mismatched = tokenize_mnli(mnli["validation_mismatched"])
    return train, val_matched, val_mismatched


# Centralized LoRA
def run_centralized():
    train_ds, val_ds, _ = load_mnli()

    with open("lora_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    lora_common = cfg["lora"]["common"]
    target_projections = lora_common["target_projections"]
    exp = cfg["lora"]["experiments"]
    run_name = exp[0]["name"]
    layer_cfg = exp[0]["layers"]

    print(f"\nRunning experiment: {run_name if USE_LORA else 'Centralised'}")
    accuracies = []

    for seed in SEEDS:
        set_seed(seed)
        selected_layers = layer_indices(layer_cfg)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3
        )

        if USE_LORA:
            target_modules = build_target_modules(
                selected_layers,
                target_projections
            )

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_common["r"],
                lora_alpha=lora_common["alpha"],
                lora_dropout=lora_common["dropout"],
                bias=lora_common["bias"],
                target_modules=target_modules
            )

            model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=f"./results/{run_name}/seed_{seed}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        best_acc = trainer.state.best_metric
        accuracies.append(best_acc)

        print(f"Seed {seed}: {best_acc:.4f}")

    print(f"{run_name} → Mean ± Std: "
            f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")


# Federated Client Training
def train_lora_client( 
    train_dataset,
    eval_dataset,
    selected_layers,
    lora_config_path,
    seed,
    num_epochs,
    initial_lora_state=None
):
    set_seed(seed)
    with open(lora_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    lora_common = cfg["lora"]["common"]

    target_modules = build_target_modules(
        selected_layers,
        lora_common["target_projections"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_common["r"],
        lora_alpha=lora_common["alpha"],
        lora_dropout=lora_common["dropout"],
        bias=lora_common["bias"],
        target_modules=target_modules
    )

    model = get_peft_model(model, lora_config)

    if initial_lora_state is not None:
        model.load_state_dict(initial_lora_state, strict=False)

    train_dataset = tokenize_mnli(train_dataset)
    eval_dataset = tokenize_mnli(eval_dataset)
    
    training_args = TrainingArguments(
        output_dir="./tmp_client",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()

    lora_state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if "lora_" in k
    }

    return lora_state, metrics

 #Evaluate aggregated global LoRA model.

def evaluate_global_lora(
    eval_dataset,
    selected_layers,
    lora_config_path,
    global_lora_state,
):
   
    # Tokenization check
    if not isinstance(eval_dataset[0]["input_ids"], torch.Tensor):
        eval_dataset = tokenize_mnli(eval_dataset)

    # Load LoRA config
    with open(lora_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    lora_common = cfg["lora"]["common"]

    target_modules = build_target_modules(
        selected_layers,
        lora_common["target_projections"]
    )

    # Build model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_common["r"],
        lora_alpha=lora_common["lora_alpha"],
        lora_dropout=lora_common["dropout"],
        bias=lora_common["bias"],
        target_modules=target_modules
    )

    model = get_peft_model(model, lora_config)

    # Load aggregated LoRA weights
    missing, unexpected = model.load_state_dict(
        global_lora_state,
        strict=False
    )

    if missing:
        print("Missing keys during global load:", missing)
    if unexpected:
        print("Unexpected keys during global load:", unexpected)

    training_args = TrainingArguments(
        output_dir="./tmp_global_eval",
        per_device_eval_batch_size=8,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    return trainer.evaluate()


if __name__ == "__main__":
    run_centralized()