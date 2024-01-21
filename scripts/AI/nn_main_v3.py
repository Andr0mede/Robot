import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

if __name__ == "__main__":
        
    class ExplorationMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(ExplorationMLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Assuming the map is a 20x20 matrix
    map_size = 20
    map_flat_size = map_size * map_size

    # Input size includes the flattened map and player position (2 values for row and column)
    input_size = map_flat_size + 2
    hidden_size = 64
    output_size = 4  # Assuming 4 possible directions (up, down, left, right)

    # Create an instance of the MLP model
    exploration_model = ExplorationMLP(input_size, hidden_size, output_size)

    # Define your optimizer and loss function
    optimizer = optim.Adam(exploration_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop (you'll need to replace this with your actual training data)
    num_episodes = 1000

    for episode in range(num_episodes):
        # Replace this with your actual training data preparation
        input_data = generateTrainingDataset(batch_size=100, flatten=True):
        target_directions = torch.randint(0, output_size, (batch_size,))

        # Forward pass
        predictions = exploration_model(input_data)

        # Compute the loss
        loss = criterion(predictions, target_directions)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode: {episode}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(exploration_model.state_dict(), 'exploration_model.pth')
