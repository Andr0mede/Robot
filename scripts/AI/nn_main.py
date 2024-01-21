
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt

from PyYel.Data.Datapoint import Datatensor

main_path = os.path.dirname(__file__)
input_data_path = main_path
output_model_path = main_path + "/Models"

hidden_layers = 256

class ConnectedNNx3(nn.Module):
    """
    A simple 3 layers fully connected neural network.
    \n
    Regression:
        Input targets: 1D vector
        Loss: MSELoss    
    Binary classification: 
        Input targets: One-hot encoded classes
        Loss: BCELoss
    Multi classification:
        Input targets: One-hot encoded classes
        Loss: CrossEntropyLoss
     \n
    Architecture:
        Linear(input_size, hidden_layers) -> ReLU, batchnorm, dropout(p) ->\n
        Linear(hidden_layers, hidden_layers//4) -> ReLU, -> Linear(hidden_layers//4, output_size)
    \n
    Args:
        in_channels:
            Images: number for color channels. 1 for grayscale, 3 for RGB...
            Other: N/A
        filters: number of filters to apply and weighten (3x3 fixed kernel size)
        hidden_layers: classifying layers size/number of neurons
        output_size: number of labels, must be equal to the length of the one hot encoded target vector.
    """

    def __init__(self, input_size, output_size, hidden_layers=128, p=0., **kwargs):
        super(ConnectedNNx3, self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)
        self.batchnorm0 = nn.BatchNorm1d(num_features=input_size)

        self.input_layer = nn.Linear(input_size, hidden_layers)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_layers)
        
        self.layer2 = nn.Linear(hidden_layers, hidden_layers//2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_layers//2)
        self.layer3 = nn.Linear(hidden_layers//2, hidden_layers//4)
        self.batchnorm3 = nn.BatchNorm1d(num_features=hidden_layers//4)
        self.layer4 = nn.Linear(hidden_layers//4, hidden_layers//8)
        self.batchnorm4 = nn.BatchNorm1d(num_features=hidden_layers//8)

        self.output_layer = nn.Linear(hidden_layers//8, output_size)

    def forward(self, x):
        
        # Enforces the right input dimension (flattening)
        x = x.view(-1, self.input_size)
        x = self.batchnorm0(x)

        # Features extraction
        # 1st layer
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        # 2nd layer
        x = self.layer2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        # 3rd layer
        x = self.layer3(x)
        x = self.relu(x)
        x = self.batchnorm3(x)

        # Classifier
        # 4th layer
        x = self.layer4(x)
        x = self.relu(x)
        # 5th layer
        x = self.output_layer(x)  

        return x
    
if __name__ == "__main__":
    from dataset_main import generateTrainingDataset

    print(f"\n> Root: {main_path}\n")

    dataset = generateTrainingDataset(batch_size=100)
    X = dataset[:, :-1]
    y = np.array([dataset[:, -1]]).astype(int).T
    num_classes = np.max(y) + 1
    y = np.squeeze(np.eye(num_classes)[y])

    # PyYel calls
    datapoint_class = Datatensor(X=X, Y=y)
    datapoint_class.runPipeline()
    # Datapoint outputs & shapes to feed to the model
    batch_size, in_channels, height, width, output_size = datapoint_class._datapointShapes(display=True)
    train_dataloader, test_dataloader = datapoint_class.getDataloaders()

if __name__ == "__main__" and "training" in sys.argv:

    input_size = in_channels

    model = ConnectedNNx3(input_size=input_size, 
                        hidden_layers=hidden_layers,
                        output_size=output_size,
                        p=0.0)
    model_name = "LayeredNN"

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train() 

    running_loss = 0.0
    losses_list = []
    #num_epochs = int(input("Number of epochs : "))    
    num_epochs = int(input("Number of training epochs: ") or 100) # User input or 100 nb if epochs
    for epoch in tqdm(range(num_epochs)):

        dataset = generateTrainingDataset()
        X = dataset[:, :-1]
        y = np.array([dataset[:, -1]]).astype(int).T
        num_classes = np.max(y) + 1
        y = np.squeeze(np.eye(num_classes)[y])

        # PyYel calls
        datapoint_class = Datatensor(X=X, Y=y)
        datapoint_class.runPipeline()
        # Datapoint outputs & shapes to feed to the model
        batch_size, in_channels, height, width, output_size = datapoint_class._datapointShapes(display=False)
        train_dataloader, test_dataloader = datapoint_class.getDataloaders()

        for inputs, targets in train_dataloader:

            optimizer.zero_grad()  
            outputs = model(inputs)  

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses_list.append(loss.item())

    print(f"Final loss: {loss.item()}")

    plt.semilogy(losses_list)
    plt.title("Training losses")
    plt.xlabel("Epochs")
    plt.ylabel("Running loss")
    plt.savefig(f"{output_model_path}/Loss_{model_name}_{num_epochs}e.png")
    # plt.show()  

    torch.save(model, f"{output_model_path}/{model_name}_{num_epochs}e.pth")
    print("Finished training, model saved")

# %% MODEL TESTING
if __name__ == "__main__" and "testing" in sys.argv:
    
    model = torch.load(f"{output_model_path}/LayeredNN_{num_epochs}e.pth")

    model.eval()
    with torch.no_grad():
        for test_features, test_labels in test_dataloader: # Tests real accuracy

            test_labels = torch.argmax(test_labels, dim=1).numpy()

            test_predicted = torch.argmax(model(test_features), dim=1).numpy()

            test_success = np.sum(np.equal(test_predicted, test_labels))
            print(f"testing accuracy: {test_success/test_predicted.shape[-1]:.4f}")

        for train_features, train_labels in train_dataloader: # Tests overfitting

            train_labels = torch.argmax(train_labels, dim=1).numpy()

            train_predicted = torch.argmax(model(train_features), dim=1).numpy()
            train_success = np.sum(np.equal(train_predicted, train_labels))
            print(f"training accuracy: {train_success/train_predicted.shape[-1]:4f}")
    print(test_predicted)


# CALLS
num_epochs=1000
def decideMove(agent_map, x_robot, y_robot):

    x_robot = np.array([x_robot])
    y_robot = np.array([y_robot])
    eval_tensor = np.concatenate([y_robot, x_robot, agent_map.flatten()])

    model = torch.load(f"{output_model_path}/LayeredNN_{num_epochs}e.pth")

    model.eval()
    with torch.no_grad():

        prediction = torch.argmax(model(torch.from_numpy(eval_tensor).float()), dim=1).numpy()[0]
        if np.random.randint(0, 2):
            prediction = np.random.randint(0, 3)

    return prediction

if __name__ == "__main__":
    agent_map = np.zeros((20, 20))
    x_robot = 1
    y_robot = 12
    print(decideMove(agent_map, x_robot, y_robot))

