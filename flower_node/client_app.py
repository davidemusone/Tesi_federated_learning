"""Client Flower per Jetson Nano."""

import argparse
import torch
import flwr as fl

from task import Net, get_device, load_data, train, test, set_seed

set_seed()

class JetsonClient(fl.client.NumPyClient):

   def __init__(self, partition_id, num_partitions, lr, local_epochs,partition_mode="iid", alpha=0.5):
       self.device = get_device()
       self.net = Net().to(self.device)
       self.lr = lr
       self.local_epochs = local_epochs
       self.trainloader, self.testloader = load_data(
           partition_id=partition_id,
           num_partitions=num_partitions,
           partition_mode=partition_mode,
           alpha=alpha
       )

       print("[CLIENT] partition_id={}, num_partitions={}, epochs={}, lr={}, device={}".format(
           partition_id, num_partitions, local_epochs, lr, self.device
       ))
       print("[CLIENT] train samples={}, test samples={}".format(
           len(self.trainloader.dataset), len(self.testloader.dataset)
       ))

   def get_parameters(self, config=None):
       return [value.detach().cpu().numpy() for value in self.net.state_dict().values()]

   def set_parameters(self, parameters):
       state_dict = self.net.state_dict()
       new_state_dict = {}

       for (key, old_value), new_value in zip(state_dict.items(), parameters):
           new_state_dict[key] = torch.tensor(new_value, dtype=old_value.dtype, device=self.device)

       self.net.load_state_dict(new_state_dict, strict=True)

   def fit(self, parameters, config):
       print("[CLIENT] Ricevuti parametri dal server")

       self.set_parameters(parameters)

       # Cast sicuro a float
       lr = float(config.get("learning_rate", self.lr))

       print("[CLIENT] Training: epochs={}, lr={}".format(self.local_epochs, lr))

       train_loss = train(
           self.net,
           self.trainloader,
           epochs=self.local_epochs,
           device=self.device,
           lr=lr
       )

       print("[CLIENT] Training completato - loss={:.4f}".format(train_loss))

       return self.get_parameters(), len(self.trainloader.dataset), {"train_loss": float(train_loss)}

   def evaluate(self, parameters, config):
       print("[CLIENT] Evaluation")

       self.set_parameters(parameters)

       loss, accuracy = test(self.net, self.testloader, device=self.device)

       print("[CLIENT] Evaluation - loss={:.4f}, accuracy={:.4f}".format(loss, accuracy))

       return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Flower Client Jetson")
   parser.add_argument("--partition-mode", type=str, default="iid", choices=["iid", "non_iid"], help="Modalità di partizionamento dei dati")
   parser.add_argument("--alpha", type=float, default=0.5, help="Parametro di concentrazione Dirichlet (solo per non_iid)")
   parser.add_argument("--partition-id", type=int, default=0)
   parser.add_argument("--num-partitions", type=int, default=4)
   parser.add_argument("--lr", type=float, default=0.01)
   parser.add_argument("--epochs", type=int, default=1)

   args = parser.parse_args()

   print("[CLIENT] Avvio Flower client: partition_id={}, num_partitions={}, epochs={}, lr={}, partition-mode={}, alpha={}".format(
       args.partition_id, args.num_partitions, args.epochs, args.lr, args.partition_mode, args.alpha
   ))

   client = JetsonClient(
       partition_id=args.partition_id,
       num_partitions=args.num_partitions,
       lr=args.lr,
       local_epochs=args.epochs,
       partition_mode=args.partition_mode,
       alpha=args.alpha
   )

   grpc_max_message_length = 1000 * 1024 * 1024

   fl.client.start_numpy_client(
       server_address="192.168.0.180:8080",
       client=client,
       grpc_max_message_length=grpc_max_message_length,
   )