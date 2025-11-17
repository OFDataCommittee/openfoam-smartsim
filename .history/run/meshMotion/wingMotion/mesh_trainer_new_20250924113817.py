import argparse
from smartredis import Client
import torch
import torch.nn as nn
from typing import Union
import numpy as np
import io
from sklearn.model_selection import train_test_split
import torch.optim as optim 

from sklearn.metrics import mean_squared_error
class EarlyStopping:
    """Early stopping with absolute threshold and patience-based logic."""

    def __init__(
        self,
        patience: int = 40,
        min_delta: float = 1.0e-4,
        model: Union[nn.Module, None] = None,
        target_loss: float = 1e-4,
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
            example_forward_input = torch.rand(2)

            # Convert the PyTorch model to TorchScript
            if self._model_script is None:
                self._model_script = torch.jit.trace(self._model, example_forward_input)
            torch.jit.save(self._model_script, buffer)
            self._model_buffer = buffer.getvalue()

        # self._model_buffer = io.BytesIO()
        # self._model.eval()
        # # Prepare a sample input
        # example_forward_input = torch.rand(2).to(device)
        # # Convert the PyTorch model to TorchScript
        # model_script = torch.jit.trace(self._model, example_forward_input)
        # # Save the TorchScript model to a buffer
        # torch.jit.save(model_script, self._model_buffer)

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

def train(num_mpi_ranks):
    client = Client()
    torch.set_default_dtype(torch.float64)
    
    # Initialize the model
    model = MLP(num_layers=3, layer_width=50, input_size=2, output_size=2, activation_fn=torch.nn.ReLU())

    # Initialize the optimizer
    learning_rate = 1e-03
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopping(
        patience=100,
        min_delta=1e-3,
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
                                                 num_mpi_ranks, 10, 1000)
        if (not points_updated):
            raise ValueError("Points dataset list not updated.")
            
        # - Poll until the displacements datasets are written by OpenFOAM
        # print (f"dataset_list_length {dataset_list_length}") # Debug info
        displacements_updated = client.poll_list_length("displacementsDatasetList", 
                                                         num_mpi_ranks, 10, 1000)
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
    
        # PYTORCH Training Loop
        loss_func = nn.MSELoss()
      
        mean_mag_displ = torch.mean(torch.norm(displ_train, dim=1))
        validation_rmse = []
        model.train()
        epochs = 100000
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
                if early_stopper(rmse_loss_val.item()):
                    print (f"RMSE {early_stopper._best_loss}, number of epochs {n_epochs}")
                    early_stopper.reset()
                    break
        # Uncomment to visualize validation RMSE
        # plt.loglog()
        # plt.title("Validation loss RMSE")
        # plt.xlabel("Epochs")
        # plt.plot(validation_rmse)
        # plt.show()

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