import argparse
from smartredis import Client
import torch
import torch.nn as nn
import numpy as np
import io
from sklearn.model_selection import train_test_split
import torch.optim as optim 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    
def sort_tensors_by_names(tensors, tensor_names):
    # Pair each tensor with its name and sort by the name
    pairs = sorted(zip(tensor_names, tensors))

    # Extract the sorted tensors
    tensor_names_sorted, tensors_sorted = zip(*pairs)

    # Convert back to list if needed
    tensor_names_sorted = list(tensor_names_sorted)
    tensors_sorted = list(tensors_sorted)

    return tensors_sorted, tensor_names_sorted

def pinn_loss(points, displ_pred):
    """
    points:      [N, 2] tensor, input coordinates (x, y)
    displ_pred:  [N, 2] tensor, predicted displacements (u_x, u_y)
    """
    assert points.shape[1] == 2 and displ_pred.shape[1] == 2, "Expecting 2D input and displacement"

    # 开启自动求导
    points.requires_grad_(True)

    u_x = displ_pred[:, 0:1]
    u_y = displ_pred[:, 1:2]

    ones = torch.ones_like(u_x)

    # ∂u_x/∂x, ∂u_x/∂y
    grad_u_x = torch.autograd.grad(u_x, points, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    dux_dx = grad_u_x[:, 0:1]
    dux_dy = grad_u_x[:, 1:2]

    # ∂u_y/∂x, ∂u_y/∂y
    grad_u_y = torch.autograd.grad(u_y, points, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    duy_dx = grad_u_y[:, 0:1]
    duy_dy = grad_u_y[:, 1:2]

    # Small strain tensor components
    eps_xx = dux_dx  # ε_xx
    eps_yy = duy_dy  # ε_yy
    eps_xy = 0.5 * (dux_dy + duy_dx)  # ε_xy = ε_yx

    # Frobenius norm squared
    eps_squared = eps_xx**2 + eps_yy**2 + 2 * eps_xy**2  # 注意：ε_xy对称出现两次

    # 总损失 = 对所有点求平均（或求和）
    pinn_loss_value = torch.mean(eps_squared)  # or torch.sum(eps_squared)

    return pinn_loss_value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def train(num_mpi_ranks):
    client = Client()
    torch.set_default_dtype(torch.float64)
    
    # Initialize the model
    model = MLP(num_layers=3, layer_width=50, input_size=2, output_size=2, activation_fn=torch.nn.SiLU()).to(device)
    log_sigma_data = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=device))
    log_sigma_phys = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=device))
    # Initialize the optimizer
    learning_rate = 1e-03
    optimizer = optim.Adam(list(model.parameters()) + [log_sigma_data, log_sigma_phys], lr=learning_rate)
    
    # Make sure all datasets are avaialble in the smartredis database.
    local_time_index = 1
    while True:
    
        print (f"Time step {local_time_index}")
          
        # Fetch datasets from SmartRedis
    
        # - Poll until the points datasets are written by OpenFOAM
        # print (f"dataset_list_length {dataset_list_length}") # Debug info
        points_updated = client.poll_list_length("pointsDatasetList", 
                                                 num_mpi_ranks, 10, 1000);
        if (not points_updated):
            raise ValueError("Points dataset list not updated.")
            
        # - Poll until the displacements datasets are written by OpenFOAM
        # print (f"dataset_list_length {dataset_list_length}") # Debug info
        displacements_updated = client.poll_list_length("displacementsDatasetList", 
                                                         num_mpi_ranks, 10, 1000);
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
      
        mean_mag_displ = torch.mean(torch.norm(displ_train, dim=1))
        validation_rmse = []
        model.train()
        epochs = 100000
        n_epochs = 0
        rmse_loss_val = 1

        data_loss_list = []
        pinn_loss_list = []
        for epoch in range(epochs):    
            # Zero the gradients
            optimizer.zero_grad()
    
            # Forward pass on the training data
            displ_pred = model(points_train)
    
            # Compute loss on the training data
            data_loss = loss_func(displ_pred, displ_train)
            p_loss = pinn_loss(points_train, displ_pred)
            # loss_train = data_loss
            # loss_train = data_loss + 0.00001 * p_loss
            loss_train = (
                torch.exp(-log_sigma_data) * data_loss + log_sigma_data +
                torch.exp(-log_sigma_phys) * p_loss + log_sigma_phys
            )
            # Backward pass and optimization
            loss_train.backward()
            optimizer.step()
            # data_loss_list.append(data_loss.item())
            # pinn_loss_list.append(p_loss.item())

            n_epochs = n_epochs + 1
            # Forward pass on the validation data, with torch.no_grad() for efficiency
            with torch.no_grad():
                displ_pred_val = model(points_val)
                mse_loss_val = loss_func(displ_pred_val, displ_val)
                rmse_loss_val = torch.sqrt(mse_loss_val)
                validation_rmse.append(rmse_loss_val)
                if (rmse_loss_val < 1e-04):
                    break
    
            # if epoch % 1000 == 0 or epoch == epochs - 1:
            #     print(f"Epoch {epoch}: Data Loss={data_loss.item():.6f}, PINN Loss={p_loss.item():.6f}, Val RMSE={rmse_loss_val.item():.6f}")
            if epoch % 10000 == 0 or epoch == epochs - 1:
                print(f"[Epoch {epoch}]")
                print(f"  Data Loss      : {data_loss.item():.6e}")
                print(f"  PINN Loss      : {p_loss.item():.6e}")
                print(f"  log_sigma_phys : {log_sigma_phys.item():.4f}")
                print(f"  weight_phys    : {torch.exp(-log_sigma_phys).item():.4e}")
                print(f"  Validation RMSE: {rmse_loss_val:.6e}")
        print (f"RMSE {validation_rmse[-1]}, number of epochs {n_epochs}")

        # Store the model into SmartRedis
        model.eval() # TEST
        # Prepare a sample input
        example_forward_input = torch.rand(2).to(device)
        # Convert the PyTorch model to TorchScript
        model_script = torch.jit.trace(model, example_forward_input)
        # Save the TorchScript model to a buffer
        model_buffer = io.BytesIO()
        torch.jit.save(model_script, model_buffer)
        # Set the model in the SmartRedis database
        print("Saving model MLP")
        client.set_model("MLP", model_buffer.getvalue(), "TORCH", "CPU")
    
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
