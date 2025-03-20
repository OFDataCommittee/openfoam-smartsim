import torch
import torch.nn as nn

# Define various RBF functions with enforced compact support
def gaussian_rbf(r):
    """Infinitely smooth Gaussian RBF, matching Wendland's implementation."""
    return torch.exp(-r**2)  # Matches WendlandLinearNetwork

def wendland_rbf(r):
    """Compactly supported Wendland C^4 RBF."""
    mask = (r < 1).float()
    rm = (1 - r).clamp(min=0.0)
    return mask * (1 + 6*r + (35/3)*r**2) * rm**6

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
    "wendland": wendland_rbf,
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
