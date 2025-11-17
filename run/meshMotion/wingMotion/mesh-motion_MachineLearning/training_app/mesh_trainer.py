import argparse
from smartredis import Client
import torch as torch
import torch.nn as nn
import numpy as np
import io
from typing import Union
from sklearn.model_selection import train_test_split
import torch.optim as optim 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
class EarlyStopping:
    """Early stopping with absolute threshold and patience-based logic."""

    def __init__(
        self,
        patience: int = 40,
        min_delta: float = 1.0e-4,
        model: Union[nn.Module, None] = None,
    ):
        self._patience = patience
        self._min_delta = min_delta
        self._model = model
        self._best_loss = float("inf")
        self._counter = 0
        self._stop = False
        self._model_buffer = None
        self._model_script = None

    def __call__(self, loss: float) -> bool:
        """Check if training should stop."""
        if loss < self._best_loss * (1.0 - self._min_delta):
            self._best_loss = loss
            self._counter = 0
            if self._model is not None:
                self.save_model()
                
        else:
            self._counter += 1
            if self._counter >= self._patience:
                self._stop = True
        return self._stop
    def reset(self):
        """Reset the early stopping state."""
        self._model.train()
        self._best_loss = float("inf")
        self._counter = 0
        self._stop = False

    def save_model(self):
        self._model.eval()
        with io.BytesIO() as buffer:
            
            if self._model_buffer:
                self._model_buffer = None

            # save the model in the buffer
            example_forward_input = torch.rand(2).to(device)

            # Convert the PyTorch model to TorchScript
            if self._model_script is None:
                self._model_script = torch.jit.trace(self._model, example_forward_input)
            torch.jit.save(self._model_script, buffer)
            self._model_buffer = buffer.getvalue()

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
    
def sort_tensors_by_names(tensors, tensor_names):
    # Pair each tensor with its name and sort by the name
    pairs = sorted(zip(tensor_names, tensors))

    # Extract the sorted tensors
    tensor_names_sorted, tensors_sorted = zip(*pairs)

    # Convert back to list if needed
    tensor_names_sorted = list(tensor_names_sorted)
    tensors_sorted = list(tensors_sorted)

    return tensors_sorted, tensor_names_sorted

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(num_mpi_ranks):
    client = Client()
    torch.set_default_dtype(torch.float64)
    
    # Initialize the model
    model = MLP(num_layers=3, layer_width=50, input_size=2, output_size=2, activation_fn=torch.nn.Tanh()).to(device)
    
    # Initialize the optimizer
    learning_rate = 1e-03
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # # L-BFGS optimizer (currently active)
    # optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100)

    early_stopper = EarlyStopping(
        patience=100,
        min_delta=1e-2,
        model=model
    )
    # Make sure all datasets are avaialble in the smartredis database.
    local_time_index = 1
    while True:    
        
        print (f"Time step {local_time_index}")
        # Fetch datasets from SmartRedis
    
        # - Poll until the points datasets are written by OpenFOAM
        # print (f"dataset_list_length {dataset_list_length}") # Debug info
        points_updated = client.poll_list_length("pointsDatasetList", 
                                                 num_mpi_ranks, 10, 5000)
        if (not points_updated):
            raise ValueError("Points dataset list not updated.")
            
        # - Poll until the displacements datasets are written by OpenFOAM
        # print (f"dataset_list_length {dataset_list_length}") # Debug info
        displacements_updated = client.poll_list_length("displacementsDatasetList", 
                                                         num_mpi_ranks, 10, 5000)
        if (not displacements_updated):
            raise ValueError("Displacements dataset list not updated.")
            
        # - Get the points and displacements datasets from SmartRedis
        points_datasets = client.get_datasets_from_list("pointsDatasetList")  
        displacements_datasets = client.get_datasets_from_list("displacementsDatasetList")
        
        # - Agglomerate all tensors from points and displacements datasets: 
        #   sort tensors by their names to ensure matching patches of same MPI ranks
        points = []
        points_names = []
        displacements = []
        displacements_names = []
    
        # Agglomerate boudary points and displacements for training.
        # TODO(TM): for mesh motion, send points_MPI_r, displacements_MPI_r and 
        #           train the MLP directly on the tensors, there is no need to 
        #           differentiate the BCs, as values are used for the training. 
        for points_dset, displs_dset in zip(points_datasets, displacements_datasets):
            points_tensor_names = points_dset.get_tensor_names()
            displs_tensor_names = displs_dset.get_tensor_names()
            for points_name,displs_name in zip(points_tensor_names,displs_tensor_names):
                patch_points = points_dset.get_tensor(points_name)
                points.append(patch_points)
                points_names.append(points_name)
    
                patch_displs = displs_dset.get_tensor(displs_name)
                displacements.append(patch_displs)
                displacements_names.append(displs_name)
                
        points, points_names = sort_tensors_by_names(points, points_names)
        displacements, displacements_names = sort_tensors_by_names(displacements, displacements_names)
        
        # - Reshape points and displacements into [N_POINTS,SPATIAL_DIMENSION] tensors
        #   This basically agglomerates data from OpenFOAM boundary patches into a list
        #   of boundary points (unstructured) and a list of respective point displacements. 
        points = torch.from_numpy(np.vstack(points))
        displacements = torch.from_numpy(np.vstack(displacements))
        
        # TODO(TM): hardcoded x,y coordinates, make the OF client store polymesh::solutionD
        #           and use solutionD non-zero values for sampling vector coordinates. 
        points = points[:, :2]
        displacements = displacements[:, :2]
    
        # Split training and validation data
        points_train, points_val, displ_train, displ_val = train_test_split(points, displacements, 
                                                                            test_size=0.2, random_state=42)
        
        points_train = points_train.clone().detach().to(device).requires_grad_(True) 
        displ_train = displ_train.to(device)
        points_val = points_val.to(device)
        displ_val = displ_val.to(device)
    
        # PYTORCH Training Loop
        loss_func = nn.MSELoss()
      
        model.train()
        epochs = 100000
        n_epochs = 0
        rmse_loss_val = 1
        
        for epoch in range(epochs):    
            # Zero the gradients
            optimizer.zero_grad()
    
            # Forward pass on the training data
            displ_pred = model(points_train)
    
            # Compute loss on the training data with annealed weight
            data_loss = loss_func(displ_pred, displ_train)

            if epoch % 50 == 0 or epoch == epochs - 1:
                print(
                    f"[Epoch {epoch}/{epochs}] "
                    f"data loss: {data_loss.item()}, "
                )
            # Backward pass and optimization
            data_loss.backward()
            optimizer.step()

        # for epoch in range(epochs):
        #     # Define closure function for L-BFGS
        #     def closure():
        #         optimizer.zero_grad()
                
        #         # Forward pass on the training data
        #         displ_pred = model(points_train)
        
        #         # Compute loss on the training data with annealed weight
        #         data_loss = loss_func(displ_pred, displ_train)
        #         p_loss = pinn_loss(points_train, displ_pred)
                
        #         # Annealed weight: start with high physics weight, gradually decrease
        #         # Physics weight decreases from 1.0 to 0.01 over training
        #         physics_weight = max(0.01, 1.0 * (1.0 - epoch / epochs))
        #         data_weight = 1.0
                
        #         loss_train = data_weight * data_loss + physics_weight * p_loss
        #         loss_train.backward()
        #         return loss_train
            
        #     # L-BFGS optimization step
        #     optimizer.step(closure)

            n_epochs = n_epochs + 1
            # Forward pass on the validation data, with torch.no_grad() for efficiency
            with torch.no_grad():
                displ_pred_val = model(points_val)
                mse_loss_val = loss_func(displ_pred_val, displ_val)
                rmse_loss_val = torch.sqrt(mse_loss_val)
                if early_stopper(rmse_loss_val.item()):
                    print(f"Training stopped at epoch {epoch}")
                    print (f"RMSE {early_stopper._best_loss}, number of epochs {n_epochs}")
                    early_stopper.reset()
                    break

        # Store the model into SmartRedis
        client.set_model("MLP", early_stopper._model_buffer, "TORCH", "CPU")
    
        # Update the model in smartredis
        client.put_tensor("model_updated", np.array([0.]))
    
        # Delete dataset lists for the next time step
        client.delete_list("pointsDatasetList")
        client.delete_list("displacementsDatasetList")

        # Update time index
        local_time_index = local_time_index + 1
        if client.poll_key("end_time_index", 10, 10):
            print ("End time reached.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for mesh motion")
    parser.add_argument("mpi_ranks", help="number of mpi ranks", type=int)
    args = parser.parse_args()

    train(args.mpi_ranks)
