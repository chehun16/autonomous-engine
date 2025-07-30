import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SimpleNN(nn.Module):
    """
    A simple neural network class for demonstration purposes.

    Attributes:
        input_size (int): The size of the input feature vector.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output feature vector.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initializes the SimpleNN model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            output_size (int): Number of output features.
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def calculate_mean_and_std(arr: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the mean and standard deviation of a numpy array.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        Tuple[float, float]: Mean and standard deviation of the array.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    return mean, std

# Example usage with comments
def main():
    """
    Example usage of the SimpleNN class and the calculate_mean_and_std function.
    """
    # Initialize the neural network
    input_size = 10
    hidden_size = 20
    output_size = 5
    model = SimpleNN(input_size, hidden_size, output_size)

    # Generate random input tensor
    x = torch.rand((4, input_size))

    # Forward pass
    output = model(x)
    print("Model output:", output)

    # Calculate mean and standard deviation of a numpy array
    arr = np.random.rand(100)
    mean, std = calculate_mean_and_std(arr)
    print(f"Mean: {mean}, Std: {std}")

if __name__ == "__main__":
    main()
