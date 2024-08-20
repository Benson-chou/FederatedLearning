import flwr as fl

strategy = fl.server.strategy.FedAvg()
# Start Flower server
fl.server.start_server(
  server_address="localhost:8080",
  config=fl.server.ServerConfig(num_rounds=1),
  strategy=strategy
)