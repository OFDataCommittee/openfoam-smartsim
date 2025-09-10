import torch
import torch.nn as nn

import torch

# Define various RBF functions with enforced compact support
def gaussian_rbf(r):
    """Infinitely smooth Gaussian RBF."""
    return torch.exp(-r**2)

def wendland_d2_c2_rbf(r):
    """
    Wendland's C^2 RBF for d=2.
    Compactly supported, continuously differentiable (C^2).
    
    Formula: (1 - r)^4_+ (4r + 1)
    """
    mask = (r < 1).float()
    rm = (1 - r).clamp(min=0.0)
    return mask * rm**4 * (4 * r + 1)

def wendland_d2_c4_rbf(r):
    """
    Wendland's C^4 RBF for d=2.
    Compactly supported, twice continuously differentiable (C^4).
    
    Formula: (1 - r)^6_+ (35r^2 + 18r + 3)
    """
    mask = (r < 1).float()
    rm = (1 - r).clamp(min=0.0)
    return mask * rm**6 * (35 * r**2 + 18 * r + 3)

def multiquadric_rbf(r):
    """Multiquadric RBF with compact support."""
    mask = (r < 1).float()
    return mask * torch.sqrt(1 + r**2)

def inverse_multiquadric_rbf(r):
    """Inverse multiquadric RBF with compact support."""
    mask = (r < 1).float()
    return mask / torch.sqrt(1 + r**2)

# Create an RBF function dictionary
rbf_dict = {
    "gaussian": gaussian_rbf,
    "wendland_d2_c2": wendland_d2_c2_rbf,
    "wendland_d2_c4": wendland_d2_c4_rbf,
    "multiquadric": multiquadric_rbf,
    "inverse_multiquadric": inverse_multiquadric_rbf
}

class RadialBasisFunctionNetwork(nn.Module):
    def __init__(self, centers, r_max, rbf_dict, rbf_type):
        """
        Generalized RBF network with user-selectable RBF functions.

        Parameters:
        centers (torch.Tensor): shape (num_centers, dimension), RBF centers.
        r_max (float): radius of compact support (applies to all RBFs).
        rbf_dict (dict): Dictionary mapping RBF type names to function implementations.
        rbf_type (str): Type of RBF function to use (must be in rbf_dict).
        """
        super().__init__()

        self.centers = centers.clone().detach()  # Fixed RBF centers
        self.r_max = r_max
        self.num_centers, self.dimension = centers.shape
        self.rbf_type = rbf_type.lower()  # Store selected RBF type as an attribute

        # Ensure rbf_type is valid
        if self.rbf_type not in rbf_dict:
            raise ValueError(f"Invalid RBF type '{self.rbf_type}'. Available options: {list(rbf_dict.keys())}")

        self.rbf_function = rbf_dict[self.rbf_type]  # Store selected RBF function

        # Trainable parameters (weights for RBFs) - match Wendland's model!
        self.weights = nn.Parameter(torch.zeros(self.num_centers))  # Initialize to zeros
        self.a0 = nn.Parameter(torch.tensor(0.0))  # Bias term initialized as 0 (like Wendland)

    def rbf(self, x):
        """
        Compute the RBF values for input x using the selected RBF function.
        """
        r = torch.cdist(x, self.centers) / self.r_max  # Compute normalized distance
        return self.rbf_function(r)  # Apply the selected RBF function

    def forward(self, x):
        """
        Forward pass: Compute RBF output.
        """
        rbf_output = self.rbf(x)
        rbf_term = rbf_output @ self.weights
        return self.a0 + rbf_term  # No polynomial correction term

    def get_rbf_type(self):
        """Return the currently selected RBF type."""
        return self.rbf_type

class RadialBasisFunctionNetworkAdaptive(nn.Module):
    def __init__(self, centers, r_max_init, rbf_dict, rbf_type):
        """
        RBF network with adaptive (trainable) support radius r_max.

        Parameters:
        centers (torch.Tensor): shape (num_centers, dimension), RBF centers.
        r_max_init (float): Initial value for radius of compact support.
        rbf_dict (dict): Dictionary mapping RBF type names to function implementations.
        rbf_type (str): Type of RBF function to use (must be in rbf_dict).
        """
        super().__init__()

        self.centers = centers.clone().detach()
        self.num_centers, self.dimension = centers.shape
        self.rbf_type = rbf_type.lower()

        if self.rbf_type not in rbf_dict:
            raise ValueError(f"Invalid RBF type '{self.rbf_type}'. Available options: {list(rbf_dict.keys())}")

        self.rbf_function = rbf_dict[self.rbf_type]

        # Trainable weights and bias
        self.weights = nn.Parameter(torch.zeros(self.num_centers))
        self.a0 = nn.Parameter(torch.tensor(0.0))

        # Adaptive r_max: log-parametrized to ensure positivity
        self.log_r_max = nn.Parameter(torch.log(torch.tensor(r_max_init)))

    def rbf(self, x):
        """
        Compute RBF values with adaptive r_max.
        """
        r_max = torch.exp(self.log_r_max)  # Ensure r_max stays positive
        r = torch.cdist(x, self.centers) / r_max
        return self.rbf_function(r)

    def forward(self, x):
        """
        Forward pass.
        """
        rbf_output = self.rbf(x)
        rbf_term = rbf_output @ self.weights
        return self.a0 + rbf_term

    def get_rbf_type(self):
        return self.rbf_type

    def get_r_max(self):
        """Return current (trainable) support radius as float."""
        return torch.exp(self.log_r_max).item()

class RadialBasisFunctionNetworkAdaptivePolynomial(nn.Module):
    def __init__(self, centers, r_max_init, rbf_dict, rbf_type):
        """
        RBF network with adaptive support radius and linear polynomial term.

        Enforces constraints: sum(w) = 0, sum(w * x) = 0, sum(w * y) = 0

        Parameters:
        centers (torch.Tensor): shape (num_centers, 2), fixed RBF centers.
        r_max_init (float): initial support radius (scalar, same for all).
        rbf_dict (dict): mapping of RBF types to functions.
        rbf_type (str): key from rbf_dict.
        """
        super().__init__()

        self.centers = centers.clone().detach()  # shape [N, 2]
        self.num_centers, self.dimension = self.centers.shape
        assert self.dimension == 2, "Only 2D supported for polynomial term."

        self.rbf_type = rbf_type.lower()
        if self.rbf_type not in rbf_dict:
            raise ValueError(f"Invalid RBF type '{self.rbf_type}'. Options: {list(rbf_dict.keys())}")

        self.rbf_function = rbf_dict[self.rbf_type]

        # Trainable RBF weights
        self.weights = nn.Parameter(torch.zeros(self.num_centers))  # w_i
        self.a0 = nn.Parameter(torch.tensor(0.0))  # constant term
        self.a1 = nn.Parameter(torch.tensor(0.0))  # x coefficient
        self.a2 = nn.Parameter(torch.tensor(0.0))  # y coefficient

        # Shared, adaptive support radius (log-param)
        self.log_r_max = nn.Parameter(torch.log(torch.tensor(r_max_init)))

        # Store polynomial constraint basis (constant, x, y at centers)
        ones = torch.ones_like(self.centers[:, 0])
        x_vals = self.centers[:, 0]
        y_vals = self.centers[:, 1]
        self.constraint_basis = torch.stack([ones, x_vals, y_vals], dim=0)  # shape [3, N]

    def rbf(self, x):
        """
        Evaluate RBF basis matrix with normalized distances.
        x: shape [M, 2]
        Returns: [M, N] matrix of φ(||x - x_i|| / r_max)
        """
        r_max = torch.exp(self.log_r_max)
        r = torch.cdist(x, self.centers) / r_max
        return self.rbf_function(r)

    def orthogonalized_weights(self):
        """
        Project weights onto the nullspace of constraint matrix.
        Enforces: w ⊥ [1, x, y]^T over centers.
        """
        C = self.constraint_basis  # shape [3, N]
        w = self.weights  # shape [N]

        # Compute projection of w onto constraint subspace
        CCt = C @ C.T              # [3 x 3]
        CCt_inv = torch.linalg.pinv(CCt)  # [3 x 3]
        projection = C.T @ (CCt_inv @ (C @ w))  # [N]
        return w - projection


    def forward(self, x):
        """
        Evaluate RBF + polynomial model at input x: shape [M, 2]
        """
        rbf_output = self.rbf(x)  # shape [M, N]
        weights_ortho = self.orthogonalized_weights()  # shape [N]
        rbf_term = rbf_output @ weights_ortho  # shape [M]

        poly_term = self.a0 + self.a1 * x[:, 0] + self.a2 * x[:, 1]  # shape [M]
        return poly_term + rbf_term

    def get_r_max(self):
        return torch.exp(self.log_r_max).item()

    def get_rbf_type(self):
        return self.rbf_type



class RadialBasisFunctionNetworkLocalAdaptive(nn.Module):
    def __init__(self, centers, r_max_init, rbf_dict, rbf_type):
        """
        RBF network with adaptive support radius per RBF center.

        Parameters:
        centers (torch.Tensor): shape (num_centers, dimension), RBF centers.
        r_max_init (float): Initial value for support radius (shared init).
        rbf_dict (dict): Dictionary mapping RBF type names to functions.
        rbf_type (str): Key to select RBF function.
        """
        super().__init__()

        self.centers = centers.clone().detach()
        self.num_centers, self.dimension = centers.shape
        self.rbf_type = rbf_type.lower()

        if self.rbf_type not in rbf_dict:
            raise ValueError(f"Invalid RBF type '{self.rbf_type}'. Options: {list(rbf_dict.keys())}")

        self.rbf_function = rbf_dict[self.rbf_type]

        # Trainable weights
        self.weights = nn.Parameter(torch.zeros(self.num_centers))
        self.a0 = nn.Parameter(torch.tensor(0.0))

        # Log of local r_max values (shape: [num_centers])
        log_r_max_init = torch.log(torch.full((self.num_centers,), r_max_init))
        self.log_r_max = nn.Parameter(log_r_max_init)

    def rbf(self, x):
        """
        Compute RBF matrix with per-center adaptive support.
        """
        # Pairwise distance: [num_points, num_centers]
        distances = torch.cdist(x, self.centers)  # shape: [M, N]

        # Per-center r_max (shape: [num_centers]) → broadcast to [M, N]
        r_max = torch.exp(self.log_r_max).unsqueeze(0)  # shape: [1, N]

        # Normalized distance
        r_normalized = distances / r_max

        return self.rbf_function(r_normalized)

    def forward(self, x):
        rbf_output = self.rbf(x)  # shape: [num_points, num_centers]
        return self.a0 + rbf_output @ self.weights

    def get_r_max(self):
        """Return current per-center r_max as 1D numpy array."""
        return torch.exp(self.log_r_max).detach().cpu().numpy()
