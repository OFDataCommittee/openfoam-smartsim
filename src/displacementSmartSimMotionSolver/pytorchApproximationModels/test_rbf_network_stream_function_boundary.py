import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os

from rbf_network import rbf_dict, RadialBasisFunctionNetwork


def velocity_u(x, y):
    return (np.sin(np.pi * x) ** 2) * np.sin(2 * np.pi * y) * np.pi

def velocity_v(x, y):
    return -np.sin(2 * np.pi * x) * (np.sin(np.pi * y) ** 2) * np.pi

def generate_boundary_points(num_rays, R, C):
    theta = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    circle_x = C[0] + R * np.cos(theta)
    circle_y = C[1] + R * np.sin(theta)
    circle_boundary = np.column_stack((circle_x, circle_y))

    outer_boundary = []
    for t in theta:
        dx, dy = np.cos(t), np.sin(t)
        intersections = []

        if dx != 0:
            for x_edge in [0.0, 1.0]:
                s = (x_edge - C[0]) / dx
                y = C[1] + s * dy
                if 0 <= y <= 1 and s > 0:
                    intersections.append([x_edge, y])
        if dy != 0:
            for y_edge in [0.0, 1.0]:
                s = (y_edge - C[1]) / dy
                x = C[0] + s * dx
                if 0 <= x <= 1 and s > 0:
                    intersections.append([x, y_edge])

        if intersections:
            dists = [np.linalg.norm(np.array(p) - np.array(C)) for p in intersections]
            outer_point = intersections[np.argmin(dists)]
            outer_boundary.append(outer_point)

    outer_boundary = np.array(outer_boundary)
    boundary_points = np.vstack([outer_boundary, circle_boundary])
    return torch.tensor(boundary_points, dtype=torch.float32)


def filter_inside_circle(points, R, C):
    distances = np.sqrt((points[:, 0] - C[0]) ** 2 + (points[:, 1] - C[1]) ** 2)
    return points[distances > R]


def visualize_velocity_field_with_mask(x, y, u, v, rbf_type, centers, title, R, C):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.quiver(x, y, u, v, scale=40)

    circle = plt.Circle(C, R, color='white', zorder=10)
    ax.add_patch(circle)

    num_centers = len(centers)
    s_max, s_min = 100, 25
    num_min, num_max = 16, 128
    s = s_max - (s_max - s_min) * (num_centers - num_min) / (num_max - num_min)
    s = max(s_min, min(s_max, s))

    centers_np = centers.numpy()
    ax.scatter(centers_np[:, 0], centers_np[:, 1], color='k', marker='x', s=s, linewidths=2, label='Centers')
    ax.legend()
    ax.set_title(f"{title} {rbf_type.upper()} {num_centers}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)

    fig_name = title.replace(" ", "-")
    plt.savefig(f"{fig_name}-rbf_type_{rbf_type}-num_centers_{num_centers}.png", dpi=200)
    plt.close(fig)

def visualize_velocity_error_norm(x, y, u_pred, v_pred, rbf_type, centers, title, R, C):
    """
    Visualize the 2-norm of the velocity error at validation points.
    """
    # Exact velocity
    u_true = velocity_u(x, y)
    v_true = velocity_v(x, y)

    error_norm = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    # Plot error as color scatter
    sc = ax.scatter(x, y, c=error_norm, cmap='magma', s=10)
    plt.colorbar(sc, ax=ax, label="||u_pred - u_true||")

    # Mask the circle area in white
    circle = plt.Circle(C, R, color='white', zorder=10)
    ax.add_patch(circle)

    # Plot centers
    num_centers = len(centers)
    s_max, s_min = 100, 25
    num_min, num_max = 16, 128
    s = s_max - (s_max - s_min) * (num_centers - num_min) / (num_max - num_min)
    s = max(s_min, min(s_max, s))
    centers_np = centers.numpy()
    ax.scatter(centers_np[:, 0], centers_np[:, 1], color='k', marker='x', s=s, linewidths=2, label='Centers')

    ax.set_title(f"{title} {rbf_type.upper()} {num_centers}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()

    fig_name = f"{title.replace(' ', '-')}-rbf_type_{rbf_type}-num_centers_{num_centers}.png"
    plt.savefig(fig_name, dpi=200)
    plt.close(fig)


def main(num_points, rbf_type):
    R = 0.15
    C = (0.5, 0.75)

    centers = generate_boundary_points(num_points, R=R, C=C)

    num_points_val = 100
    x_val = np.linspace(0, 1, num_points_val)
    y_val = np.linspace(0, 1, num_points_val)
    X_val, Y_val = np.meshgrid(x_val, y_val)
    xy_val = np.column_stack((X_val.flatten(), Y_val.flatten()))
    xy_val_filtered = filter_inside_circle(xy_val, R=R, C=C)

    # Training data (u, v)
    x_train = centers
    u_train = torch.tensor(velocity_u(centers[:, 0], centers[:, 1]), dtype=torch.float32)
    v_train = torch.tensor(velocity_v(centers[:, 0], centers[:, 1]), dtype=torch.float32)

    # Fit two RBF models: one for u, one for v
    r_max = 3 / num_points

    def train_component_model(y_train):
        model = RadialBasisFunctionNetwork(x_train, r_max, rbf_dict, rbf_type=rbf_type)
        optimizer = optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.MSELoss()
        best_loss = float("inf")
        best_model_state = None
        stop_loss = 1e-6
        epochs = 4000

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = model.state_dict().copy()

            if loss.item() < stop_loss:
                break

        if best_model_state:
            model.load_state_dict(best_model_state)
        model.eval()
        return model

    model_u = train_component_model(u_train)
    model_v = train_component_model(v_train)

    # Predict velocities at validation points
    with torch.no_grad():
        x_val_torch = torch.tensor(xy_val_filtered, dtype=torch.float32)
        u_pred = model_u(x_val_torch).numpy()
        v_pred = model_v(x_val_torch).numpy()

    # Visualize velocity field
    visualize_velocity_field_with_mask(
        xy_val_filtered[:, 0], xy_val_filtered[:, 1], u_pred, v_pred,
        rbf_type, centers, title="Velocity Field", R=R, C=C
    )

    # Visualize 2-norm error
    visualize_velocity_error_norm(
        xy_val_filtered[:, 0], xy_val_filtered[:, 1], u_pred, v_pred,
        rbf_type, centers, title="Velocity Error Norm", R=R, C=C
    )

    # Save mean/max error if desired
    u_true = velocity_u(xy_val_filtered[:, 0], xy_val_filtered[:, 1])
    v_true = velocity_v(xy_val_filtered[:, 0], xy_val_filtered[:, 1])
    err_u = np.abs(u_pred - u_true) / (np.max(np.abs(u_true)) + 1e-12)
    err_v = np.abs(v_pred - v_true) / (np.max(np.abs(v_true)) + 1e-12)

    csv_filename = "velocity_validation.csv"
    header = ["model_rbf_type", "num_points", "r_max", "err_mean_u", "err_max_u", "err_mean_v", "err_max_v"]
    data = [rbf_type, num_points, r_max, np.mean(err_u), np.max(err_u), np.mean(err_v), np.max(err_v)]

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)

    print(f"Appended to {csv_filename}: {data}")


if __name__ == "__main__":
    for rbf_type in ["gaussian", "wendland_d2_c4"]:
        for num_points in [16, 32, 64, 128]:
            main(num_points, rbf_type)
