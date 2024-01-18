
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt

main_path = os.path.abspath("")
input_data_path = main_path
output_model_path = main_path 

# print(f"\n> Root: {main_path}\n")

# dataset = generateTrainingDataset()
# X = dataset[:, :-1]
# y = np.array([dataset[:, -1]]).T
# print(y.shape)
# num_classes = np.max(y) + 1
# y = np.squeeze(np.eye(num_classes)[y])

# # PyYel calls
# datapoint_class = Datapoint(X=X, y=y)
# datapoint_class.runPipeline()
# # Datapoint outputs & shapes to feed to the model
# batch_size, in_channels, height, width, output_size = datapoint_class._datapointShapes(display=True)
# train_dataloader, test_dataloader = datapoint_class.getDataloaders()


class LayeredNN(nn.Module):
    def __init__(self, input_size, output_size=1, sigma=1, hidden_size=64):
        super(LayeredNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sigma = sigma

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.1)
        self.softmax = nn.Softmax(dim=1)

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_size)
        
        self.layer1 = nn.Linear(hidden_size, hidden_size//2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_size//2)
        self.layer2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_size//4)
        self.layer3 = nn.Linear(hidden_size//4, hidden_size//8)
        self.batchnorm3 = nn.BatchNorm1d(num_features=hidden_size//8)

        self.output_layer = nn.Linear(hidden_size//8, output_size)

            
    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)
        # x = self.batchnorm2(x)

        x = self.layer3(x)
        x = self.relu(x)

        x = self.dropout(x)
        
        x = self.output_layer(x)  
        # x = self.softmax(x)

        return x


# if __name__ == "__main__" and "training" in sys.argv:

#     hidden_size = 64
#     input_size = in_channels

#     model = LayeredNN(input_size=input_size, 
#                         hidden_size=hidden_size,
#                         output_size=output_size)
#     model_name = "LayeredNN"

#     criterion = nn.CrossEntropyLoss()
#     # criterion = nn.BCEWithLogitsLoss()

#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     model.train() 

#     running_loss = 0.0
#     losses_list = []
#     #num_epochs = int(input("Number of epochs : "))    
#     num_epochs = int(input("Number of training epochs: ") or 100) # User input or 100 nb if epochs
#     for epoch in tqdm(range(num_epochs)):

#         dataset = generateTrainingDataset()
#         X = dataset[:, :-1]
#         y = np.array([dataset[:, -1]]).T
#         num_classes = np.max(y) + 1
#         y = np.squeeze(np.eye(num_classes)[y])

#         # PyYel calls
#         datapoint_class = Datapoint(X=X, y=y)
#         datapoint_class.runPipeline()
#         # Datapoint outputs & shapes to feed to the model
#         batch_size, in_channels, height, width, output_size = datapoint_class._datapointShapes(display=False)
#         train_dataloader, test_dataloader = datapoint_class.getDataloaders()

#         for inputs, targets in train_dataloader:

#             optimizer.zero_grad()  
#             outputs = model(inputs)  

#             loss = criterion(outputs, targets)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             losses_list.append(loss.item())

#     print(f"Final loss: {loss.item()}")

#     plt.semilogy(losses_list)
#     plt.title("Training losses")
#     plt.xlabel("Epochs")
#     plt.ylabel("Running loss")
#     plt.savefig(f"{output_model_path}/Loss_{model_name}_{num_epochs}e.png")
#     plt.show()

#     torch.save(model.state_dict(), f"{output_model_path}/{model_name}_{num_epochs}e.pth")
#     print("Finished training, model saved")

# # %% MODEL TESTING
# if __name__ == "__main__" and "testing" in sys.argv:
#     try:
#         model = LayeredNN(input_size=in_channels, output_size=output_size)
#         model.load_state_dict(torch.load(f"{output_model_path}/LayeredNN_{num_epochs}e.pth"))
#         model_loaded = "Currently trained model used "
#     except:
#         num_epochs = int(input("Number of training epochs: ") or 100) # User input or 100 nb if epochs
#         model = LayeredNN(input_size=in_channels, output_size=output_size)
#         model.load_state_dict(torch.load(f"{output_model_path}/LayeredNN_{num_epochs}e.pth"))
#         model_loaded = f"{num_epochs}e model loaded"

#     # Loading pre trained model if available
#     if (False or ("pretrained" in sys.argv)):
#         try:
#             model.load_state_dict(torch.load(f"{output_model_path}/CNN_10e.pth"))
#             model_loaded = "10 epochs pre trained model loaded"
#         except:
#             None
#         try:
#             model.load_state_dict(torch.load(f"{output_model_path}/CNN_100e.pth"))
#             model_loaded = "100 epochs pre trained model loaded"
#         except:
#             None
#         try:
#             model.load_state_dict(torch.load(f"{output_model_path}/CNN_1000e.pth"))
#             model_loaded = "1k pre trained model loaded"
#         except:
#             None
#         try:
#             model.load_state_dict(torch.load(f"{output_model_path}/CNN_10000e.pth"))
#             model_loaded = "10k pre trained model loaded"
#         except:
#             None
#         try:
#             model.load_state_dict()
#             model.load_state_dict(torch.load(f"{output_model_path}/CNN_20000e.pth"))
#             model_loaded = "20k pre trained model loaded"
#         except:
#             None
#     print(model_loaded)

#     model.eval()
#     with torch.no_grad():
#         for test_features, test_labels in test_dataloader: # Tests real accuracy

#             test_labels = torch.argmax(test_labels, dim=1).numpy()

#             test_predicted = torch.argmax(model(test_features), dim=1).numpy()

#             test_success = np.sum(np.equal(test_predicted, test_labels))
#             print(f"testing accuracy: {test_success/test_predicted.shape[-1]:.4f}")

#         for train_features, train_labels in train_dataloader: # Tests overfitting

#             train_labels = torch.argmax(train_labels, dim=1).numpy()

#             train_predicted = torch.argmax(model(train_features), dim=1).numpy()
#             train_success = np.sum(np.equal(train_predicted, train_labels))
#             print(f"training accuracy: {train_success/train_predicted.shape[-1]:4f}")

#     # predict_values, pred_counts = np.unique(test_predicted, return_counts=True)
#     # labels_values, test_counts = np.unique(test_labels, return_counts=True)
#     # print(pred_counts)
#     # print(test_counts)
#     print(test_predicted)

def decideMove(agent_map, x_robot, y_robot):

    x_robot = np.array([x_robot])
    y_robot = np.array([y_robot])
    eval_tensor = np.concatenate(y_robot, x_robot, agent_map.flatten())
    in_channels=len(eval_tensor)
    output_size=4

    try:
        model = LayeredNN(input_size=in_channels, output_size=output_size)
        model.load_state_dict(torch.load(f"{output_model_path}/LayeredNN_1000e.pth"))
        # model_loaded = "Currently trained model used "
    except:
        raise ImportError("Can't find/load/fit NN weights")

    model.eval()
    with torch.no_grad():

        prediction = torch.argmax(model(eval_tensor), dim=1).numpy()

    return prediction        
