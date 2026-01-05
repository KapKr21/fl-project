from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
import numpy as np
import evaluate
from peft.tuners.lora import LoraLayer
import random
import torch




NUM_LAYERS = 12  # roberta-base
model_name = "roberta-base" #125 million parameters
tokenizer = AutoTokenizer.from_pretrained(model_name)
USE_LORA = True 

def data_processing():
    mnli = load_dataset("glue", "mnli")
    # print(mnli["train"][0])

    train_ds = mnli["train"]
    val_matched_ds = mnli["validation_matched"]
    val_mismatched_ds = mnli["validation_mismatched"]

    # print("Train size:", len(train_ds))
    # print("Validation (matched):", len(val_matched_ds))
    # print("Validation (mismatched):", len(val_mismatched_ds))

    return train_ds, val_matched_ds, val_mismatched_ds

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def apply_tokenizer(train_ds, val_matched_ds, val_mismatched_ds, tokenizer):

    def tokenize_and_keep_labels(batch):
        tokenized = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
        tokenized["labels"] = batch["label"]
        return tokenized

    train_tokenized = train_ds.map(
        tokenize_and_keep_labels,
        batched=True,
        remove_columns=train_ds.column_names
    )

    val_matched_tokenized = val_matched_ds.map(
        tokenize_and_keep_labels,
        batched=True,
        remove_columns=val_matched_ds.column_names
    )

    val_mismatched_tokenized = val_mismatched_ds.map(
        tokenize_and_keep_labels,
        batched=True,
        remove_columns=val_mismatched_ds.column_names
    )

    train_tokenized.set_format("torch")
    val_matched_tokenized.set_format("torch")
    val_mismatched_tokenized.set_format("torch")

    return train_tokenized, val_matched_tokenized, val_mismatched_tokenized


#Declaring training and evaluation data
train_ds, val_matched_ds, val_mismatched_ds = data_processing()

#Applying tokenizer to the training and evaluation data
train_tokenized,val_matched_tokenized,val_mismatched_tokenized = apply_tokenizer(train_ds, val_matched_ds, val_mismatched_ds,tokenizer)

# print(train_tokenized[0])


def layer_indices(layer_cfg):
    mode = layer_cfg["mode"]

    if mode == "explicit":
        selected_layers = layer_cfg["indices"]

    elif mode == "top_k":
        k = layer_cfg["top_k"]
        selected_layers = list(range(NUM_LAYERS - k, NUM_LAYERS))

    elif mode == "range":
        start, end = layer_cfg["range"]
        selected_layers = list(range(start, end + 1))

    elif mode == "all":
        selected_layers = list(range(NUM_LAYERS))

    else:
        raise ValueError(f"Unknown layer selection mode: {mode}")

    return selected_layers


tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3  #Three labels for classification. 
)


#To Do : Layer selection is done using the lora_config.yaml file here 
with open("lora_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

lora_common = cfg["lora"]["common"]

r = lora_common["r"]
alpha = lora_common["alpha"]
dropout = lora_common["dropout"]
bias = lora_common["bias"]
target_projections = lora_common["target_projections"]

#Call to select the indices 
# selected_layers = layer_indices(layer_cfg)

# Loading compute metricx
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# # Create target_modules for applying LoRa onto specific layers 
# target_modules = []
# for layer_id in selected_layers:
#     for proj in target_projections:
#         target_modules.append(
#             f"roberta.encoder.layer.{layer_id}.attention.self.{proj}"
#         )


# # print("LoRA will be applied to the following modules:")
# # for m in target_modules:
# #     print(m)


# lora_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS, #sequence classification (Works well with MNLI type of tasks - usually sequence based classifications)
#     r=r,
#     lora_alpha=alpha,
#     lora_dropout=dropout,
#     bias=bias,
#     target_modules=target_modules
# )


# # Apply LoRA to attention Query and Value projections
# if USE_LORA:
#     model = get_peft_model(model, lora_config)

# #Checking applied modules through the model variable
# print("Modules with LoRA adapters:\n")
# for name, module in model.named_modules():
#     if isinstance(module, LoraLayer):
#         print(name)

# run_name = "lora_alternate_odd_layers"

# #Trainig Arguments 
# training_args = TrainingArguments(
#     output_dir=f"./results/{run_name}",
#     logging_dir=f"./logs/{run_name}",
#     eval_strategy="epoch",
#     save_strategy="epoch",

#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,

#     num_train_epochs=3,
#     weight_decay=0.01,

#     logging_steps=100,

#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",

#     fp16=True,               
#     report_to="none"         
# )


# #Trainer construction
# trainer = Trainer(
#     model=model,
#     args=training_args,

#     train_dataset=train_tokenized,
#     eval_dataset=val_matched_tokenized,

#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# # trainer.train()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEEDS = [42, 123, 999]

for exp in cfg["lora"]["experiments"]:

    run_name = exp["name"]
    layer_cfg = exp["layers"]

    print(f"\nRunning experiment: {run_name}")

    accuracies = []

    for seed in SEEDS:
        print(f"Seed {seed}")
        set_seed(seed)

        selected_layers = layer_indices(layer_cfg)

        target_modules = []
        for layer_id in selected_layers:
            for proj in target_projections:
                target_modules.append(
                    f"roberta.encoder.layer.{layer_id}.attention.self.{proj}"
                )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3
        )

        if USE_LORA:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias=bias,
                target_modules=target_modules
            )
            model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=f"./results/{run_name}/seed_{seed}",
            logging_dir=f"./logs/{run_name}/seed_{seed}",
            eval_strategy="epoch", 
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_matched_tokenized,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()

        best_acc = trainer.state.best_metric
        accuracies.append(best_acc)

        print(f"Accuracy: {best_acc:.4f}")

    print(f"\n{run_name} results: {accuracies}")
    print(f"Mean ± Std: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

