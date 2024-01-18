
from PyYel.Networks.Compiler import Trainer, Tester, Loader
from PyYel.Networks.Models import CNNx2, CNNx3
from PyYel.Data.Datapoint import Datatensor, YelDataset, YelDatapoint

from dataset_main import generateTrainingDataset

import os
import numpy as np
import sys
import torch

# %% DEVELOPMENT

if __name__ == "__main__":
        
    X, Y = generateTrainingDataset(batch_size=1000, flatten=False)

    print(X.shape, Y.shape)

    X = YelDatapoint(X)
    X.reshape()
    X = X.getModifiedData()
    print(X.shape)

    Y = YelDatapoint(Y)
    Y.oneHotEncode()
    Y = Y.getModifiedData()

    dataset = YelDataset(X=X, Y=Y).getModifiedDataset()

    datatensor = Datatensor(X=X, Y=Y)
    datatensor.runPipeline(dtype='float', delta=1e-3)
    train_dataloader, test_dataloader = datatensor.getDataloaders()

if __name__ == "__main__" and "training" in sys.argv:

    model = CNNx3(in_channels=1, filters=8, hidden_layers=64, input_size=22, output_size=4)
    model_name = "CNNx3"

    trainer = Trainer(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, num_epochs=2000,
            model_name=model_name, output_path=os.path.join(os.path.dirname(__file__), "Models"))
    trainer.runPipeline()

if __name__ == "__main__" and "testing" in sys.argv:

    loader = Loader(model_name="CNNx3_2000e", input_path=os.path.join(os.path.dirname(__file__), "Models"))
    model = loader.getModel()

    tester = Tester(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    tester.runPipeline()


# %% IMPORTABLE

loader = Loader(model_name="CNNx3_1000e", input_path=os.path.join(os.path.dirname(__file__), "Models"))
model = loader.getModel()
def decideMove(agent_map, x_robot, y_robot):

    x_robot = np.array([x_robot])
    y_robot = np.array([y_robot])
    array = np.pad(agent_map, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    array[x_robot, 0] = 1
    array[x_robot, -1] = 1
    array[y_robot, 0] = 1
    array[y_robot, -1] = 1

    output = model(torch.from_numpy(np.array([array])).float().unsqueeze(0))
    prediction = torch.argmax(output).numpy()

    return prediction        

if __name__ == "__main__":
    agent_map = np.zeros((20, 20))
    agent_map[4, 12] = 1
    x_robot = 1
    y_robot = 12
    print(decideMove(agent_map, x_robot, y_robot))
