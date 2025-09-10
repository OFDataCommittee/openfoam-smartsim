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

def train(num_mpi_ranks):
    client = Client()
    torch.set_default_dtype(torch.float64)

    # Read the solution direction from a database
    dimension = int(client.get_tensor("solution_dim"))

    print (f"Solution dimension = {dimension}.")
    
    # Initialize the model
    model = MLP(
        num_layers=3, 
        layer_width=50, 
        input_size=dimension, 
        output_size=dimension, 
        activation_fn=torch.nn.ReLU()
    )

    # Initialize the optimizer
    learning_rate = 1e-03
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

        # Split training and validation data 
        points_train, points_val, displ_train, displ_val = train_test_split(
            points,
            displacements,
            test_size=0.2,
            random_state=42
        )

        # Convert to torch.Tensor 
        points_train = torch.from_numpy(points_train).to(torch.float64)
        points_val   = torch.from_numpy(points_val).to(torch.float64)
        displ_train  = torch.from_numpy(displ_train).to(torch.float64)
        displ_val    = torch.from_numpy(displ_val).to(torch.float64)
    
        loss_func = nn.MSELoss()
      
        mean_mag_displ = torch.mean(torch.norm(displ_train, dim=1))
        validation_rmse = []
        model.train()
        epochs = 2000
        n_epochs = 0
        rmse_loss_val = 1

        for epoch in range(epochs):    
            # Zero the gradients
            optimizer.zero_grad()
    
            # Forward pass on the training data
            displ_pred = model(points_train)
    
            # Compute loss on the training data
            loss_train = loss_func(displ_pred, displ_train)
    
            # Backward pass and optimization
            loss_train.backward()
            optimizer.step()

            n_epochs = n_epochs + 1
            # Forward pass on the validation data, with torch.no_grad() for efficiency
            with torch.no_grad():
                displ_pred_val = model(points_val)
                mse_loss_val = loss_func(displ_pred_val, displ_val)
                rmse_loss_val = torch.sqrt(mse_loss_val)
                validation_rmse.append(rmse_loss_val)
                if (mse_loss_val < 1e-04):
                    break
    
        print (f"RMSE {validation_rmse[-1]}, number of epochs {n_epochs}")

        # Uncomment to visualize validation RMSE
        plt.loglog()
        plt.title("Validation loss RMSE")
        plt.xlabel("Epochs")
        plt.plot(validation_rmse)
        plt.savefig(f"validation_rmse_{epoch:04d}.png")
    
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
    args = parser.parse_args()

    train(args.mpi_ranks)
