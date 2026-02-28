# Adaptive Aggregation for Federated Fine-Tuning with LoRA

This repository implements the FedIT framework, exploring the intersection of Low-Rank Adaptation (LoRA) and Federated Learning (FL) using RoBERTa-base. The project specifically investigates "aggregation noise" and mathematical inconsistencies in standard Federated Averaging (FedAvg) when applied to non-linear low-rank subspaces.
Key Features;
   1. Centralized Baseline: High-fidelity replication of original LoRA [1] results.
   2. 2. Layer-wise Ablation: Tools to identify optimal Transformer layer subsets (e.g., Middle 6) for adaptation.
   3. FedIT Framework: Decentralized fine-tuning using standard FedAvg.
   4. Overlap-Aware Aggregation: Support for Architectural Heterogeneity, allowing clients to participate with disparate layer configurations.
   5.  Non-IID Simulation: Data heterogeneity support for GLUE tasks (MNLI, SST-2).The project is designed to explore how different LoRA layer selection strategies affect federated learning performance on GLUE tasks.

---

## Repository Structure

├── base_lora.py              # Centralized LoRA implementation & Layer ablation

├── federated_lora.py         # Main entry point for FedIT experiments

├── flora_overlap_aggregate.py # Overlap-aware FedAvg logic

├── data_utils.py             # Non-IID data partitioning (MNLI/SST-2)

├── flora_utils.py            # Model surgery & parameter extraction utilities

├── lora_config.yaml          # Hyperparameters (Rank r=8, Alpha, etc.)

├── requirements.txt          # Dependency list

└── README.md


---
## File Descriptions
base_lora.py

Implements the centralized LoRA fine-tuning. As noted in our report, this script was used to establish that tuning the Middle 6 layers achieves an optimal balance, reaching ~83% accuracy on MNLI with only 0.74M parameters.
flora_overlap_aggregate.py

This is the core of our Scenario 3 implementation. It handles the logic for:
    1. Overlapping Layers: Averaging parameters shared by multiple clients.
    2. Outlier Layers: Preserving unique layers from specialized clients without aggregation, mitigating subspace noise.

federated_lora.py
The orchestration engine for the three scenarios defined in the report:
    1. Scenario 1: 4 Clients, Full 12-layer LoRA (Homogeneous).
    2. Scenario 2: 4 Clients, Middle 6 layers (Homogeneous Partial).
    3. Scenario 3: 2 Clients, Heterogeneous Layer subsets (Middle 6 vs. Middle 4).

## Experimental Scenarios

Layer selection is currently **hard-coded inside `federated_lora.py`**.

---

### Scenario 1 – 4 Clients, All LoRA Layers

- Number of clients: 4
- Each client trains **all LoRA layers**
- Non-IID data split

This serves as the federated baseline.

---

### Scenario 2 – 4 Clients, Middle Layers Only

- Number of clients: 4
- Each client trains **only the middle transformer layers**
- Layers are manually specified in the script

Purpose:
Evaluate performance when only partial LoRA layers are trained.

---

### Scenario 3 – 2 Clients, Heterogeneous Layer Training

- Number of clients: 2

Layer assignment:
- Client 1 → Middle 6 layers
- Client 2 → Middle 4 layers

Purpose:
Simulate heterogeneous client capabilities and overlapping layer participation.

---

## Technical Details
### Aggregation Noise
Our experiments empirically prove that element-wise averaging of A and B matrices independently is mathematically inconsistent. This project serves as a testbed for observing the "structural collapse" of adapters in decentralized settings.

### Communication Efficiency
By utilizing the Middle 6 layer configuration, this framework reduces communication overhead by 99.4% compared to Full Fine-Tuning (FFT), reducing the per-round payload from ~500MB to ~2.96MB.

---

Installation & Usage
Install Dependencies:

    Bash
    pip install -r requirements.txt

Run Centralized Ablation:

    Bash
    python base_lora.py

Run Federated Simulation:

    Bash
    python federated_lora.py 

Notes:
1. Layer selection is currently hard-coded.
2. Designed for experimental research purposes.
3. Aggregation assumes consistent LoRA parameter naming.
4. Results are stored in CSV format for further analysis.
