import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

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

def visualize_psi(x, y, psi_values, title="Stream Function", centers=None):
    plt.figure(figsize=(6, 6))
    plt.contourf(x, y, psi_values, levels=20, cmap='viridis')
    plt.colorbar(label='ψ')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    
    # Plot centers if provided
    if centers is not None:
        centers_np = centers.numpy()  # Convert from torch tensor to numpy
        plt.scatter(centers_np[:, 0], centers_np[:, 1], color='white', marker='x', s=100, linewidths=2, label='Centers')
        plt.legend()
    
    fig_name = title.replace(" ", "-")
    plt.savefig(f"{fig_name}.png", dpi=200)

def visualize_velocity_field(x, y, u, v, title="Velocity Field"):
    plt.figure(figsize=(6, 6))
    plt.quiver(x, y, u, v)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    fig_name = title.replace(" ", "-")
    plt.savefig(f"{fig_name}.png", dpi=200)

def generate_centers(num_centers):
    """
    Generate grid-based RBF centers within the domain [0, 1] x [0, 1].
    """
    x = np.linspace(0, 1, num_centers)
    y = np.linspace(0, 1, num_centers)
    X, Y = np.meshgrid(x, y)
    centers = np.vstack([X.ravel(), Y.ravel()]).T
    return torch.tensor(centers, dtype=torch.float32)

def main():
    # Generate training data
    num_points = 10
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
    r_max = 0.1 
    smoothness = 4  # C^4 smoothness

    # Initialize model
    model = WendlandLinearNetwork(centers, r_max, smoothness)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    # Training loop
    epochs = 2000
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
    num_points = 100
    x_val = np.linspace(0, 1, num_points)
    y_val = np.linspace(0, 1, num_points)
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
    #psi_actual = psi(X_val, Y_val)

    # Visualize actual and predicted stream functions
    visualize_psi(X_val, Y_val, psi_val, title="Actual Stream Function")
    visualize_psi(X_val, Y_val, psi_pred, title="Predicted Stream Function")

    visualize_psi(X_val, Y_val, np.abs(psi_pred - psi_val), 
                  title="Stream Function Approximation Error", 
                  centers=centers)

    # Compute and visualize actual and predicted velocity fields
    u_val, v_val = compute_velocity(X_val, Y_val, psi_val)
    visualize_velocity_field(X_val, Y_val, u_val, v_val, title="Actual Velocity Field")

    u_pred, v_pred = compute_velocity(X_val, Y_val, psi_pred)
    visualize_velocity_field(X_val, Y_val, u_pred, v_pred, title="Predicted Velocity Field")

if __name__ == "__main__":
    main()