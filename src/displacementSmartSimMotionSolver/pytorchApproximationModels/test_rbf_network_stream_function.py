import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import csv 
import pandas as pd
import os

from rbf_network import rbf_dict, RadialBasisFunctionNetwork 

def psi(x, y):
    """
    Compute the stream function ψ(x, y) for the given domain.
    """
    return np.sin(np.pi * x)**2 * np.sin(np.pi * y)**2

def compute_velocity(x, y, psi_values):
    """
    Compute velocity field (u, v) from stream function ψ.
    """
    dy, dx = np.gradient(psi_values, y[:, 0], x[0, :])
    u = dy  # u = ∂ψ/∂y
    v = -dx  # v = -∂ψ/∂x
    return u, v

def visualize_psi(x, y, psi_values, rbf_type, centers, title):
    plt.figure(figsize=(6, 6))
    plt.contourf(x, y, psi_values, levels=20, cmap='viridis')
    plt.colorbar(label='ψ')
    plt.title(title + f"-rbf_type_{rbf_type}-num_centers_{len(centers)}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    
    # Plot centers 
    centers_np = centers.numpy()  # Convert from torch tensor to numpy
    plt.scatter(centers_np[:, 0], centers_np[:, 1], color='white', marker='x', s=100, linewidths=2, label='Centers')
    plt.legend()
    
    fig_name = title.replace(" ", "-")
    plt.savefig(f"{fig_name}-rbf_type_{rbf_type}-num_centers_{len(centers)}.png", dpi=200)

def visualize_velocity_field(x, y, u, v, rbf_type, num_centers, title="Velocity"):
    plt.figure(figsize=(6, 6))
    plt.quiver(x, y, u, v)
    plt.title(f"{title}-{rbf_type}-n_centers_{num_centers}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    fig_name = title.replace(" ", "-")
    plt.savefig(f"{fig_name}-rbf_type_{rbf_type}-num_centers_{num_centers}.png", dpi=200)

def generate_centers(num_centers):
    """
    Generate grid-based RBF centers within the domain [0, 1] x [0, 1].
    """
    x = np.linspace(0, 1, num_centers)
    y = np.linspace(0, 1, num_centers)
    X, Y = np.meshgrid(x, y)
    centers = np.vstack([X.ravel(), Y.ravel()]).T
    return torch.tensor(centers, dtype=torch.float32)

def estimate_convergence_order(csv_filename):
    """
    Opens a CSV file containing numerical convergence results and estimates the
    convergence order for each error column using log-log error reduction.

    The last row's convergence order is set equal to the second-to-last row.
    
    Parameters:
    csv_filename (str): The path to the CSV file to process.
    """
    # Load the CSV file
    df = pd.read_csv(csv_filename)

    # Ensure required columns exist
    required_columns = {"point_dist", "err_mean", "err_max"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns in CSV: {required_columns - set(df.columns)}")

    # List of error columns to process
    error_columns = ["err_mean", "err_max"]

    for error_col in error_columns:
        convergence_orders = []  # Store convergence orders for this error type

        for i in range(len(df) - 1):  # Iterate up to the second-to-last row
            h_coarse, h_fine = df.iloc[i]["point_dist"], df.iloc[i + 1]["point_dist"]
            err_coarse, err_fine = df.iloc[i][error_col], df.iloc[i + 1][error_col]

            if err_coarse > 0 and err_fine > 0:  # Avoid log errors due to zero or negative values
                p = np.log(err_coarse / err_fine) / np.log(h_coarse / h_fine)
                convergence_orders.append(p)
            else:
                convergence_orders.append(np.nan)

        # Ensure last row gets the same convergence order as the previous row
        convergence_orders.append(convergence_orders[-1] if len(convergence_orders) > 0 else np.nan)

        # Add convergence order column to DataFrame
        df[f"{error_col}_convergence_order"] = convergence_orders

    # Save the updated CSV file
    df.to_csv(csv_filename, index=False)

    print(f"Updated {csv_filename} with convergence orders for {error_columns}.")

def main(num_points, rbf_type):
    # Generate training data
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    X, Y = np.meshgrid(x, y)
    xy_train = np.column_stack((X.flatten(), Y.flatten()))
    psi_train = psi(xy_train[:, 0], xy_train[:, 1])

    # Convert training data to torch tensors
    x_train = torch.tensor(xy_train, dtype=torch.float32)
    y_train = torch.tensor(psi_train, dtype=torch.float32)

    # Generate centers
    centers = x_train 
    print(centers.shape)

    # Gaussian 3d-order support
    #r_max = 2.5 / num_points 
    r_max = 2.5 / num_points 

    # Initialize model
    model = RadialBasisFunctionNetwork(centers, r_max, rbf_dict, rbf_type=rbf_type)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.MSELoss()

    # Training loop
    epochs = 4000
    best_loss = float("inf")  # Initialize best loss to a large value
    best_model_state = None  # Store best model state
    stop_loss = 1e-08

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # Save the model if it has the lowest loss so far
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()  # Copy best model state

        # Early stopping criterion
        if loss.item() < stop_loss:
            print(f"Stopping early at epoch {epoch + 1} due to reaching loss {loss.item()} < {stop_loss}")
            break

        # Print progress every 50 epochs
        if epoch == 1 or (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.14f}, Best Loss: {best_loss:.14f}')

    # Restore the best model state if training didn't reach convergence
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with loss: {best_loss:.14f}")

    # Save the best model to file
    torch.save(best_model_state, "best_rbf_model.pth")
    print("Best model saved as 'best_rbf_model.pth'.")

    # Generate validation data
    num_points_val = 100
    x_val = np.linspace(0, 1, num_points_val)
    y_val = np.linspace(0, 1, num_points_val)
    X_val, Y_val = np.meshgrid(x_val, y_val)
    xy_val = np.column_stack((X_val.flatten(), Y_val.flatten()))
    psi_val = psi(X_val, Y_val)

    # Evaluate predictions at validation points
    model.eval()
    with torch.no_grad():
        x_val = torch.tensor(xy_val, dtype=torch.float32)
        pred = model(x_val).numpy()

    # Reshape for visualization
    psi_pred = pred.reshape(X_val.shape)

    # Visualize actual and predicted stream functions
    visualize_psi(X_val, Y_val, psi_val, rbf_type, 
                  centers, title="Actual Stream Function")
    visualize_psi(X_val, Y_val, psi_pred, rbf_type,
                  centers, title="Predicted Stream Function")

    err_val = np.abs(psi_pred - psi_val) / np.max(psi_val)
    visualize_psi(X_val, Y_val, err_val, rbf_type, centers,
                  title="Stream Function Relative Error")

    # Define the filename
    csv_filename = "stream_function_validation.csv"

    # Define the header and the values to be appended
    header = ["model_rbf_type", "num_points", "support_radius", 
              "point_dist", "err_mean", "err_max"]

    data = [model.rbf_type, num_points, r_max, 1.0 / num_points, 
            np.mean(err_val), np.max(err_val)]

    # Check if file exists
    file_exists = os.path.isfile(csv_filename)

    # Open file in append mode
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file doesn't exist, write the header first
        if not file_exists:
            writer.writerow(header)

        # Append the data row
        writer.writerow(data)

    print(f"Appended to {csv_filename}: {data}")

    # Compute and visualize actual and predicted velocity fields
    u_val, v_val = compute_velocity(X_val, Y_val, psi_val)
    visualize_velocity_field(X_val, Y_val, u_val, v_val, rbf_type, num_points,
                             title="Velocity Field")

    u_pred, v_pred = compute_velocity(X_val, Y_val, psi_pred)
    visualize_velocity_field(X_val, Y_val, u_pred, v_pred, rbf_type, num_points,
                             title="Predicted Velocity Field")

if __name__ == "__main__":

    # Run the parameter study 
    for rbf_type in ["gaussian"]:
        for num_points in [4,8,16,32]:
            main(num_points, rbf_type)

    # Estimate convergence order
    estimate_convergence_order("stream_function_validation.csv")