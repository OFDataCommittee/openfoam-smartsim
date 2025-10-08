import argparse
import torch
import numpy as np
import io
import torch.optim as optim

from matplotlib import pyplot as plt
from smartredis import Client

from MLP import MLP, MLPTrainer


def train(args):
    client = Client()
    torch.set_default_dtype(torch.float64)

    # Read the solution direction from a database
    dimension = int(client.get_tensor("solution_dim")[0])

    print (f"Solution dimension = {dimension}.")
    # Initialize the model
    if args.model_name == "mlp":
        model = MLP(
            input_size=dimension,
            output_size=dimension,
            num_layers=3,
            layer_width=10,
            activation_fn=torch.nn.ELU()
        )
        trainer = MLPTrainer(model, args.radius_power)

    data_ready = client.poll_key("points", 1, 10000)
    points = client.get_tensor("points")
    interior_points = np.vstack([client.get_tensor(f"points_MPI_{i}" for i in range(4))])
    X = torch.from_numpy(points).to(torch.float64)
    # Make sure all datasets are avaialble in the smartredis database.

    epochs = 5000
    iteration = 1
    while True:

        print (f"Iteration {iteration}")

        data_ready = client.poll_key("data_ready", 1, 10000)
        if (not data_ready):
            raise RuntimeError("Data not found in SmartRedis; aborting training.")

        displacements = client.get_tensor("displacements")
        interior_points = client.get_tensor
        client.delete_tensor("data_ready")

        y = torch.from_numpy(displacements).to(torch.float64)


        validation_rmse = []
        n_epochs = 0

        for epoch in range(epochs):
            loss, model = trainer.training_step(X, y)
            if trainer.converged():
                break

        print(f"MSE {loss.item()}, number of epochs {epoch}", flush=True)
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
    parser.add_argument("model_name",
                        help="which model to use to calculate interior displacements",
                        choices=["mlp"],
                        type=str
    )
    args = parser.parse_args()

    train(args)
