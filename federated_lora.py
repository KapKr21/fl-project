import csv
from base_LoRa import train_lora_client,evaluate_global_lora
from data_utils import split_glue_non_iid
from flora_overlap_aggregate import overlap_aware_aggregate
from flora_utils import flora_aggregate,flora_aggregate_weighted


# Configuration
MODEL_NAME = "roberta-base"
NUM_CLIENTS = 4
NUM_ROUNDS = 10
LOCAL_EPOCHS = 5
SEEDS = [42, 123, 999]

LORA_CONFIG_PATH = "lora_config.yaml"
SCENARIO = 1

# Client 
class Client:
    def __init__(self, client_id, train_data, eval_data):
        self.client_id = client_id
        self.train_data = train_data
        self.eval_data = eval_data

    def local_train(self, global_lora_state,selected_layers):
        lora_state, metrics = train_lora_client(
            train_dataset=self.train_data,
            eval_dataset=self.eval_data,
            lora_config_path=LORA_CONFIG_PATH,
            seed = SEEDS[self.client_id % len(SEEDS)],
            num_epochs=LOCAL_EPOCHS,
            initial_lora_state=global_lora_state,
            selected_layers = selected_layers 
        )
        return lora_state, metrics,selected_layers


# Federated training loop
def run_federated_flora():
    print("Starting Scenario 1: FLORA with LoRA on all layers")
    
    # Non-IID data split
    if SCENARIO == 1:
        client_splits = split_glue_non_iid(task_name='mnli',num_clients=NUM_CLIENTS,dominant_frac=0.7)
    elif SCENARIO == 3:
        client_splits = split_glue_non_iid(task_name='sst2',num_clients=NUM_CLIENTS,dominant_frac=0.7)
    
    results = []  

    # Create clients
    clients = [
        Client(
            client_id=i,
            train_data=split["train"],
            eval_data=split["eval"],
        )
        for i, split in enumerate(client_splits)
    ]

    global_lora_state = None
    if SCENARIO == 1:
        selected_layers = list(range(12))
    elif SCENARIO == 3:
        CLIENT_LAYER_MAP = {
            0: [3, 4, 5, 6, 7, 8],
            1: [4, 5, 6, 7]
        }

    for round_id in range(NUM_ROUNDS):
        print(f"\n===== Communication Round {round_id} =====")
        client_lora_states = []
        client_updates = []
        for client in clients:
            print(f"\nClient {client.client_id} local training...")
            if SCENARIO == 1:
                lora_state, metrics = client.local_train(global_lora_state, selected_layers)
                client_lora_states.append(lora_state)

            elif SCENARIO == 3:
                selected_layers = CLIENT_LAYER_MAP[client.client_id]
                lora_state, metrics, trained_layers = client.local_train(
                    global_lora_state,
                    selected_layers
                )
            client_updates.append({
                "client_id": client.client_id,
                "lora_state": lora_state,
                "layers": trained_layers,
                "num_samples": len(client.train_data),
            })
        
            print(f"Client {client.client_id} eval accuracy: {metrics.get('eval_accuracy')}")
            results.append({
                "round": round_id,
                "client_id": client.client_id,
                "eval_accuracy": metrics["eval_accuracy"]
            })

            # Check if LoRA params exist
            print(f"Client {client.client_id} LoRA params: {len(lora_state)}")

        client_sizes = [len(client.train_data) for client in clients]
        # Aggregation
        if SCENARIO == 1:
            global_lora_state = flora_aggregate(client_lora_states)

        elif SCENARIO == 3:
            global_lora_state = overlap_aware_aggregate(client_updates)

        print("Num global params:", len(global_lora_state))
        
        global_metrics = evaluate_global_lora(
            model_name=MODEL_NAME,
            eval_dataset=client_splits[0]["eval"], 
            lora_config_path=LORA_CONFIG_PATH,
            global_lora_state=global_lora_state,
            selected_layers= selected_layers
        )
        
        round_accs = [
            r["eval_accuracy"]
            for r in results
            if r["round"] == round_id and r["client_id"] != "mean"
        ]

        mean_acc = sum(round_accs) / len(round_accs)

        results.append({
            "round": round_id,
            "client_id": "mean",
            "eval_accuracy": mean_acc,
            "global_acc":global_metrics["eval_accuracy"]
        })
        
        print(f"Round {round_id} mean accuracy: {mean_acc:.4f}")
    
    with open("scenario1_vanilla_avg_10_rounds_r16_2e_5_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "client_id", "eval_accuracy","global_acc"]
        )
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    run_federated_flora()
