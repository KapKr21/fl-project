def flora_aggregate_weighted(client_lora_states, client_sizes):
    total = sum(client_sizes)
    agg_state = {}

    for key in client_lora_states[0].keys():
        agg_state[key] = sum(
            (client_sizes[i] / total) * client_lora_states[i][key]
            for i in range(len(client_lora_states))
        )

    return agg_state



def flora_aggregate(client_lora_states):
    agg_state = {}

    for key in client_lora_states[0].keys():
        agg_state[key] = sum(
            client_state[key] for client_state in client_lora_states
        ) / len(client_lora_states)

    return agg_state

