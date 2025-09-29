import argparse
from smartredis import Client
import torch
import torch.nn as nn
import numpy as np
import io
from sklearn.model_selection import train_test_split
import torch.optim as optim
import time
from typing import Tuple, Union
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error

class MLP(nn.Module):
    def __init__(self, num_layers, layer_width, input_size, output_size, activation_fn):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, layer_width))
        layers.append(activation_fn)

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(activation_fn)

        layers.append(nn.Linear(layer_width, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def loss_weighted_center(y_true, y_pred, weights, weights_power):
    weights_normed = torch.pow(weights, weights_power)
    weights_normed = weights_normed/torch.sum(weights_normed)

    return torch.sum(torch.sum((y_true-y_pred)**2, dim=1)*weights_normed)


def train(args):
    client = Client()
    torch.set_default_dtype(torch.float64)

    # Read the solution direction from a database
    dimension = int(client.get_tensor("solution_dim"))

    print (f"Solution dimension = {dimension}.")

    # Initialize the model
    model = MLP(
        num_layers=3,
        layer_width=10,
        input_size=dimension,
        output_size=dimension,
        activation_fn=torch.nn.ELU()
    )

    # Initialize the optimizer
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Make sure all datasets are avaialble in the smartredis database.
    iteration = 1
    while True:

        print (f"Iteration {iteration}")

        data_ready = client.poll_key("data_ready", 1, 10000)
        if (not data_ready):
            raise RuntimeError("Data not found in SmartRedis; aborting training.")

        points = client.get_tensor("points")
        displacements = client.get_tensor("displacements")

        client.delete_tensor("data_ready")

        X = torch.from_numpy(points).to(torch.float64)
        y = torch.from_numpy(displacements).to(torch.float64)

        # Find the center of the shape as the average of all the points on the inner boundary
        r = torch.sqrt(torch.sum(X**2, dim=1))
        inner = r < 5
        center = torch.mean(X[inner], dim=0)

        dist = torch.sqrt(torch.sum((X-center)**2, dim=1))
        wts = dist/torch.sum(dist)

        validation_rmse = []
        model.train()
        epochs = 5000
        n_epochs = 0

        for epoch in range(epochs):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass on the training data
            displ_pred = model(X)

            # Compute loss on the training data
            loss_train = loss_weighted_center(displ_pred, y, wts, args.radius_power)

            if (loss_train < 5e-05):
                break

            # Backward pass and optimization
            loss_train.backward()
            optimizer.step()

            n_epochs = n_epochs + 1

        print (f"MSE {loss_train.item()}, number of epochs {n_epochs}", flush=True)
        np.savez(
            f"data_{iteration:02d}.npz",
            points=points,
            displacements=displacements,
        )

        # Uncomment to visualize validation RMSE
        plt.loglog()
        plt.title("Validation loss RMSE")
        plt.xlabel("Epochs")
        plt.plot(validation_rmse)
        plt.savefig(f"validation_rmse_{iteration:04d}.png")

        # Store the model into SmartRedis
        # Put the model in evaluation mode.
        model.eval() # TEST
        # Prepare a sample input
        example_forward_input = torch.rand(dimension)
        # Convert the PyTorch model to TorchScript
        model_script = torch.jit.trace(model, example_forward_input)
        # Save the TorchScript model to a buffer
        model_buffer = io.BytesIO()
        torch.jit.save(model_script, model_buffer)
        # Set the model in the SmartRedis database
        print("Saving model")
        client.set_model("model", model_buffer.getvalue(), "TORCH", "CPU")
        client.put_tensor("model_ready", np.array([0]))

        # Increase CFD+ML iteration
        iteration = iteration + 1

        # Check final iteration index and break
        if client.poll_key("final_iteration", 10, 10):
           print ("final iteration reached.")
           break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for mesh motion")
    parser.add_argument("mpi_ranks", help="number of mpi ranks", type=int)
    parser.add_argument("radius_power", help="power law to weight losses", type=float)
    args = parser.parse_args()

    train(args)
