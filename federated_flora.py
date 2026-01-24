# federated_flora.py
  
import csv
from base_LoRa import train_lora_client
from data_utils import split_mnli_non_iid
from flora_utils import flora_aggregate


# -------------------------
# Configuration
# -------------------------

MODEL_NAME = "roberta-base"
NUM_CLIENTS = 2
NUM_ROUNDS = 3
LOCAL_EPOCHS = 3
SEEDS = [42, 123, 999]

LORA_CONFIG_PATH = "lora_config.yaml"


# Client 
class Client:
    def __init__(self, client_id, train_data, eval_data):
        self.client_id = client_id
        self.train_data = train_data
        self.eval_data = eval_data

    def local_train(self, global_lora_state):
        lora_state, metrics = train_lora_client(
            model_name=MODEL_NAME,
            train_dataset=self.train_data,
            eval_dataset=self.eval_data,
            lora_config_path=LORA_CONFIG_PATH,
            seed=SEEDS[self.client_id],
            num_epochs=LOCAL_EPOCHS,
            initial_lora_state=global_lora_state,
        )
        return lora_state, metrics


# Federated training loop
def run_federated_flora():
    # print("Starting Scenario 1: FLORA with LoRA on all layers")
    print("Starting Scenario 2: FLORA with LoRA on middle layers")

    # Non-IID data split
    client_splits = split_mnli_non_iid(num_clients=NUM_CLIENTS)

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
    
    for round_id in range(NUM_ROUNDS):
        print(f"\n===== Communication Round {round_id} =====")
        client_lora_states = []

        for client in clients:
            print(f"\nClient {client.client_id} local training...")
            lora_state, metrics = client.local_train(global_lora_state)

            print(f"Client {client.client_id} eval accuracy: {metrics.get('eval_accuracy')}")
            results.append({
                "round": round_id,
                "client_id": client.client_id,
                "eval_accuracy": metrics["eval_accuracy"]
            })

            # Check if LoRA params exist
            print(f"Client {client.client_id} LoRA params: {len(lora_state)}")

            client_lora_states.append(lora_state)
            
            
        # Aggregation
        global_lora_state = flora_aggregate(client_lora_states)
        round_accs = [
            r["eval_accuracy"]
            for r in results
            if r["round"] == round_id and r["client_id"] != "mean"
        ]

        mean_acc = sum(round_accs) / len(round_accs)

        results.append({
            "round": round_id,
            "client_id": "mean",
            "eval_accuracy": mean_acc
        })
        
        print(f"Round {round_id} mean accuracy: {mean_acc:.4f}")
    
    # with open("scenario1_results.csv", "w", newline="") as f:
    with open("scenario2_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "client_id", "eval_accuracy"]
        )
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    run_federated_flora()
