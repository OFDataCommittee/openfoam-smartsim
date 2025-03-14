import torch
import torch.nn as nn
import math

class WendlandLinearNetwork(nn.Module):
    def __init__(self, centers, r_max, smoothness):
        """
        Initialize the Wendland RBF network with linear polynomial terms.

        Parameters:
        centers (torch.Tensor): shape (num_centers, dimension) RBF centers.
        r_max (float): radius of compact support.
        smoothness (int): even integer specifying desired smoothness (C^smoothness).
        """
        super().__init__()

        if smoothness != 4:
            raise NotImplementedError("Only smoothness=4 (C⁴) is currently implemented explicitly.")

        self.centers = centers.clone().detach()
        self.r_max = r_max
        self.smoothness = smoothness
        self.k = smoothness // 2
        self.num_centers, self.dimension = centers.shape

        # Trainable parameters (explicitly initialized!)
        self.weights = nn.Parameter(torch.zeros(self.num_centers))
        self.a0 = nn.Parameter(torch.tensor(0.0))
        self.a = nn.Parameter(torch.zeros(self.dimension))

    def rbf(self, x):
        """
        Correct explicit Wendland polynomial RBF (d=2, k=2).
        """
        r = torch.cdist(x, self.centers) / self.r_max
        mask = (r < 1).float()
        rm = (1 - r).clamp(min=0.0)
        
        phi = (1 + 6*r + (35/3)*r**2) * (mask * (1 - r).pow(6))
        phi = mask * (1 + 6 * r + (35/3) * r**2) * rm**6
        phi = phi * mask  # strictly enforce compact support
        
        return phi

    def forward(self, x):
        """
        Forward pass: Wendland RBF with explicit linear polynomial extension.
        """
        rbf_output = self.rbf(x)
        rbf_term = rbf_output @ self.weights
        linear_term = x @ self.a
        return self.a0 + rbf_term + linear_term