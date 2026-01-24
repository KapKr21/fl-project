def flora_aggregate(client_lora_states):
    agg_state = {}

    for key in client_lora_states[0].keys():
        agg_state[key] = sum(
            client_state[key] for client_state in client_lora_states
        ) / len(client_lora_states)

    return agg_state
