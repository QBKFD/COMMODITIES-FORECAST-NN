import argparse
import json
from cfnn.train_nn import train_nn_model

def main():
    parser = argparse.ArgumentParser(description="Train NeuralForecast NN Model")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    train_nn_model(config)

if __name__ == "__main__":
    main()
