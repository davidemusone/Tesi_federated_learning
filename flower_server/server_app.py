"""Server coordinatore Flower da eseguire sul Mac."""

import argparse
import torch
import flwr as fl
from task import Net, get_device, load_centralized_dataset, test


def get_initial_parameters():
    """Estrae i pesi iniziali dalla rete PyTorch e li converte nel formato Flower."""
    net = Net()
    weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return fl.common.weights_to_parameters(weights)


def get_evaluate_fn():
    """Restituisce la funzione di valutazione globale per il Server."""
    device = get_device()
    
    try:
        testloader = load_centralized_dataset(batch_size=64)
    except Exception as e:
        print(f"Attenzione: Impossibile caricare il testset globale ({e}). Valutazione server disabilitata.")
        testloader = None

    current_round = 0

    def evaluate(weights, *args, **kwargs):
        nonlocal current_round

        if testloader is None:
            return 0.0, {"accuracy": 0.0}

        if isinstance(weights, fl.common.Parameters):
            weights = fl.common.parameters_to_weights(weights)

        net = Net().to(device)
        
        params_dict = zip(net.state_dict().keys(), weights)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(net, testloader, device=device)
        
        print(
            f"\n[SERVER - Round {current_round}] Valutazione Globale -> "
            f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n"
        )
        
        current_round += 1
        
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--num-clients", type=int, default=4, help="Numero di nodi/client attesi per il round")
    parser.add_argument("--rounds", type=int, default=10, help="Numero di round di addestramento")
    args = parser.parse_args()

    NUM_ROUNDS = args.rounds
    NUM_CLIENTS = args.num_clients

    print(f"Inizializzazione Server Flower 0.18.0 (Target Client: {NUM_CLIENTS}, Round: {NUM_ROUNDS})...")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=NUM_CLIENTS,        # Impostato dinamicamente
        min_available_clients=NUM_CLIENTS,  # Impostato dinamicamente
        min_eval_clients=NUM_CLIENTS,       # Impostato dinamicamente
        initial_parameters=get_initial_parameters(),
        eval_fn=get_evaluate_fn(),
    )

    grpc_max_message_length = 1000 * 1024 * 1024

    print(f"Server pronto! Avvio del socket gRPC su 0.0.0.0:8080 per {NUM_ROUNDS} rounds...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": NUM_ROUNDS},
        strategy=strategy,
        grpc_max_message_length=grpc_max_message_length,
    )