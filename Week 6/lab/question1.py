import numpy as np
import matplotlib.pyplot as plt

# In the visualization, 1 represents a blank space and 0 represents a dark square

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))  # Initialize the weight matrix

    def train(self, patterns):
        """Train the Hopfield network using the provided binary patterns."""
        for pattern in patterns:
            # Convert pattern to bipolar form (-1, 1)
            bipolar_pattern = 2 * pattern - 1
            self.weights += np.outer(bipolar_pattern, bipolar_pattern)
        
        # Remove self-connections by zeroing the diagonal
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, iterations=5):
        """Retrieve a pattern from the network using asynchronous updates."""
        current_state = pattern.copy()
        for _ in range(iterations):
            for i in range(self.size):
                net_input = np.dot(self.weights[i], current_state)
                current_state[i] = 1 if net_input > 0 else -1
        return (current_state + 1) // 2  # Convert back to binary (0, 1)

    def energy(self, state):
        """Compute the energy of the current state."""
        bipolar_state = 2 * state - 1
        return -0.5 * np.dot(bipolar_state.T, np.dot(self.weights, bipolar_state))

# Initialize a 10x10 Hopfield network
network_size = 100  # For a 10x10 grid, the size is 100
hopfield_net = HopfieldNetwork(network_size)

# Generate binary patterns (each pattern is a 10x10 grid flattened into a 1D array)
binary_patterns = [
    np.random.randint(0, 2, network_size),  # Random binary pattern
    np.random.randint(0, 2, network_size),  # Another random binary pattern
]

# Train the network with the patterns
hopfield_net.train(binary_patterns)

# Test the network with a pattern
test_pattern = binary_patterns[0].copy()
# Add noise to the test pattern
noise_indices = np.random.choice(network_size, network_size // 10, replace=False)
test_pattern[noise_indices] = 1 - test_pattern[noise_indices]  # Flip bits

# Recall the pattern from the network
recalled_pattern = hopfield_net.recall(test_pattern)

# Function to display and print the patterns
def display_and_print_pattern(pattern, title):
    """Display the pattern as an image and print it as a matrix."""
    pattern_matrix = pattern.reshape(10, 10)
    print(f"{title} (Binary Matrix):")
    print(pattern_matrix)
    print("\n")
    plt.imshow(pattern_matrix, cmap="gray", interpolation="nearest")
    plt.title(title)
    plt.axis("off")

# Plot and print the patterns
plt.figure(figsize=(10, 4))

# Original pattern
plt.subplot(1, 3, 1)
display_and_print_pattern(binary_patterns[0], "Original Pattern")

# Noisy input pattern
plt.subplot(1, 3, 2)
display_and_print_pattern(test_pattern, "Noisy Input Pattern")

# Recalled pattern
plt.subplot(1, 3, 3)
display_and_print_pattern(recalled_pattern, "Recalled Pattern")

plt.tight_layout()
plt.show()