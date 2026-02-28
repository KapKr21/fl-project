from collections import defaultdict
import torch

def overlap_aware_aggregate(client_updates):
    aggregated_state = {}
    param_buckets = defaultdict(list)

    # 1. Group tensors by parameter name
    for update in client_updates:
        for name, tensor in update["lora_state"].items():
            param_buckets[name].append(
                (tensor, update["num_samples"])
            )

    # 2. Aggregate per parameter
    for param_name, tensors_and_weights in param_buckets.items():
        tensors = [t for t, _ in tensors_and_weights]
        weights = [w for _, w in tensors_and_weights]

        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()

        # weighted sum
        aggregated_state[param_name] = sum(
            w * t for w, t in zip(weights, tensors)
        )

    return aggregated_state
