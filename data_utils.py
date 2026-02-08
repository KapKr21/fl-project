from datasets import load_dataset, Dataset
from collections import defaultdict
import random

def split_sst2_non_iid(num_clients=4, seed=42, dominant_frac=0.7):
    random.seed(seed)

    dataset = load_dataset("glue", "sst2")
    train_ds = dataset["train"]
    eval_ds = dataset["validation"]

    # Group examples by label
    label_buckets = defaultdict(list)
    for ex in train_ds:
        label_buckets[ex["label"]].append(ex)

    # Shuffle each label bucket
    for label in label_buckets:
        random.shuffle(label_buckets[label])

    clients = [[] for _ in range(num_clients)]

    labels = list(label_buckets.keys())  # [0, 1]
    num_labels = len(labels)

    # Assign dominant label per client (round-robin)
    for client_id in range(num_clients):
        dominant_label = labels[client_id % num_labels]

        for label in labels:
            bucket = label_buckets[label]

            if label == dominant_label:
                take = int(dominant_frac * len(bucket) / num_clients)
            else:
                take = int((1 - dominant_frac) * len(bucket) / ((num_labels - 1) * num_clients))

            clients[client_id].extend(bucket[:take])
            del bucket[:take]

    return [
        {
            "train": Dataset.from_list(client_data),
            "eval": eval_ds,
        }
        for client_data in clients
    ]
