import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import csv 
import pandas as pd
import os

from rbf_network import WendlandLinearNetwork

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

def visualize_psi(x, y, psi_values, centers, title="Stream Function"):
    plt.figure(figsize=(6, 6))
    plt.contourf(x, y, psi_values, levels=20, cmap='viridis')
    plt.colorbar(label='ψ')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    
    # Plot centers 
    centers_np = centers.numpy()  # Convert from torch tensor to numpy
    plt.scatter(centers_np[:, 0], centers_np[:, 1], color='white', marker='x', s=100, linewidths=2, label='Centers')
    plt.legend()
    
    fig_name = title.replace(" ", "-")
    plt.savefig(f"{fig_name}-num_centers-{len(centers)}.png", dpi=200)

def visualize_velocity_field(x, y, u, v, num_centers, title="Velocity Field"):
    plt.figure(figsize=(6, 6))
    plt.quiver(x, y, u, v)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    fig_name = title.replace(" ", "-")
    plt.savefig(f"{fig_name}-num_centers{num_centers}.png", dpi=200)

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
    convergence order for each row using log-log error reduction.

    The last row's convergence order is set equal to the second-to-last row.
    
    Parameters:
    csv_filename (str): The path to the CSV file to process.
    """
    # Load the CSV file
    df = pd.read_csv(csv_filename)

    # Ensure required columns exist
    required_columns = {"point_dist", "err_validation"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns in CSV: {required_columns - set(df.columns)}")

    # Compute convergence order using log-log slope formula
    convergence_orders = []

    for i in range(len(df) - 1):  # Iterate up to the second-to-last row
        h_coarse, h_fine = df.iloc[i]["point_dist"], df.iloc[i + 1]["point_dist"]
        err_coarse, err_fine = df.iloc[i]["err_validation"], df.iloc[i + 1]["err_validation"]

        if err_coarse > 0 and err_fine > 0:  # Avoid log errors due to zero or negative values
            p = np.log(err_coarse / err_fine) / np.log(h_coarse / h_fine)
            convergence_orders.append(p)
        else:
            convergence_orders.append(np.nan)

    # Ensure last row gets the same convergence order as the previous row
    convergence_orders.append(convergence_orders[-1] if len(convergence_orders) > 0 else np.nan)

    # Add convergence order column
    df["error_convergence_order"] = convergence_orders

    # Save the updated CSV file
    df.to_csv(csv_filename, index=False)

    print(f"Updated {csv_filename} with convergence orders.")

def main(num_points):
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
    # centers = generate_centers(32).clone().detach()
    centers = x_train 
    print(centers.shape)
    r_max = 2.5 / num_points 
    smoothness = 4  # C^4 smoothness

    # Initialize model
    model = WendlandLinearNetwork(centers, r_max, smoothness)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    # Training loop
    epochs = 4000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if loss.item() < 1e-12:
            print(f"Stopping early at epoch {epoch + 1} due to reaching loss {loss.item()} < 1e-05")
            break

        if ((epoch == 1) or ((epoch + 1) % 50 == 0)):
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.14f}')

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
    visualize_psi(X_val, Y_val, psi_val, centers, title="Actual Stream Function")
    visualize_psi(X_val, Y_val, psi_pred, centers, title="Predicted Stream Function")

    err_val = np.abs(psi_pred - psi_val) / np.max(psi_val)
    visualize_psi(X_val, Y_val, err_val, centers,
                  title="Stream Function Relative Approximation Error")

    # Define the filename
    csv_filename = "stream_function_validation.csv"

    # Define the header and the values to be appended
    header = ["num_points", "point_dist", "r_max", "err_validation"]
    data = [num_points, 1.0 / num_points, r_max, np.mean(err_val)]

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
    visualize_velocity_field(X_val, Y_val, u_val, v_val, num_points,
                             title="Actual Velocity Field")

    u_pred, v_pred = compute_velocity(X_val, Y_val, psi_pred)
    visualize_velocity_field(X_val, Y_val, u_pred, v_pred, num_points,
                             title="Predicted Velocity Field")

if __name__ == "__main__":

    # Run mesh convergence study
    for num_points in [4,8,16,32]:
        main(num_points)

    # Estimate convergence order
    estimate_convergence_order("stream_function_validation.csv")