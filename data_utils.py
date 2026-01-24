from datasets import load_dataset
from collections import defaultdict
import random
from datasets import Dataset

def split_mnli_non_iid(num_clients=2, seed=42):
    assert num_clients == 2

    random.seed(seed)

    dataset = load_dataset("glue", "mnli")
    train_ds = dataset["train"]
    eval_ds = dataset["validation_matched"]

    # Group by label
    buckets = defaultdict(list)
    for ex in train_ds:
        buckets[ex["label"]].append(ex)

    for k in buckets:
        random.shuffle(buckets[k])

    client_0 = []
    client_1 = []

    # Entailment - More entailment data for client 0 
    split = int(0.8 * len(buckets[0]))
    client_0.extend(buckets[0][:split])
    client_1.extend(buckets[0][split:])

    # Neutral - about the  same for both
    split = int(0.5 * len(buckets[1]))
    client_0.extend(buckets[1][:split])
    client_1.extend(buckets[1][split:])

    # Contradiction - More entailment data for client 1
    split = int(0.2 * len(buckets[2]))
    client_0.extend(buckets[2][:split])
    client_1.extend(buckets[2][split:])


    return [
        {
            "train": Dataset.from_list(client_0),
            "eval": Dataset.from_list(eval_ds) if isinstance(eval_ds, list) else eval_ds,
        },
        {
            "train": Dataset.from_list(client_1),
            "eval": Dataset.from_list(eval_ds) if isinstance(eval_ds, list) else eval_ds,
        },
    ]

