
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
from PyYel.Networks.Models import ConnectedNNx3
from PyYel.Networks.Compiler import Loader, Trainer, Tester

from AI.dataset_main import generateTrainingDataset

main_path = os.path.abspath("")
input_data_path = main_path
output_model_path = main_path + "/Models"

hidden_layers = 512

if __name__ == "__main__":
        
    print(f"\n> Root: {main_path}\n")

    dataset = generateTrainingDataset(batch_size=100)
    X = dataset[:, :-1]
    y = np.array([dataset[:, -1]]).T
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
        y = np.array([dataset[:, -1]]).T
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
    plt.show()  

    torch.save(model.state_dict(), f"{output_model_path}/{model_name}_{num_epochs}e.pth")
    print("Finished training, model saved")

# %% MODEL TESTING
if __name__ == "__main__" and "testing" in sys.argv:
    try:
        model = ConnectedNNx3(input_size=in_channels, output_size=output_size, hidden_layers=hidden_layers)
        model.load_state_dict(torch.load(f"{output_model_path}/LayeredNN_{num_epochs}e.pth"))
        model_loaded = "Currently trained model used "
    except:
        num_epochs = int(input("Number of training epochs: ") or 100) # User input or 100 nb if epochs
        model = ConnectedNNx3(input_size=in_channels, output_size=output_size, hidden_layers=hidden_layers)
        model.load_state_dict(torch.load(f"{output_model_path}/LayeredNN_{num_epochs}e.pth"))
        model_loaded = f"{num_epochs}e model loaded"

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

    # predict_values, pred_counts = np.unique(test_predicted, return_counts=True)
    # labels_values, test_counts = np.unique(test_labels, return_counts=True)
    # print(pred_counts)
    # print(test_counts)
    print(test_predicted)

def decideMove(agent_map, x_robot, y_robot):

    x_robot = np.array([x_robot])
    y_robot = np.array([y_robot])
    eval_tensor = np.concatenate([y_robot, x_robot, agent_map.flatten()])
    in_channels = len(eval_tensor)
    output_size=4

    try:
        model = ConnectedNNx3(input_size=in_channels, output_size=output_size, hidden_layers=hidden_layers)
        model.load_state_dict(torch.load(os.path.dirname(__file__)+"/Models/LayeredNN_10000e.pth"))
        # model_name = f"LayeredNN_10000e.pth"
        # model = Loader(model_name=model_name, 
        #                input_path=os.path.join(os.path.abspath(''), f"Models")).getModel()
        # model_loaded = "Currently trained model used "
    except:
        raise ImportError("Can't find/load/fit NN weights")

    model.eval()
    with torch.no_grad():

        prediction = torch.argmax(model(torch.from_numpy(eval_tensor).float()), dim=1).numpy()
        if np.random.randint(0, 2):
            prediction = np.random.randint(0, 3)
        print(prediction)
    return prediction        

if __name__ == "__main__":
    agent_map = np.zeros((20, 20))
    x_robot = 1
    y_robot = 12
    print(decideMove(agent_map, x_robot, y_robot)[0])

