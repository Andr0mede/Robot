from PyYel.datapoint import Datapoint
from PyYel.compiler import Trainer, Tester, Loader
from PyYel.NNmodels import CNN

import numpy as np
import os
from dataset_main import generateTrainingDataset

main_path = os.path.abspath("")
input_data_path = main_path
output_model_path = main_path + "/Models"
print(f"\n> Root: {main_path}\n")

X, y = generateTrainingDataset(batch_size=5000, flatten=False)
print(X.shape, y.shape)
num_classes = np.max(y) + 1
y = np.squeeze(np.eye(num_classes)[y])[:, 1:]

# PyYel calls
datapoint_class = Datapoint(X=X, y=y)
datapoint_class.runPipeline()
# Datapoint outputs & shapes to feed to the model
batch_size, in_channels, height, width, output_size = datapoint_class._datapointShapes(display=True)
train_dataloader, test_dataloader = datapoint_class.getDataloaders()
kwargs = datapoint_class.getKwargs()

# Model choice with parameters from Datapoint
model = CNN(filters=4, hidden_layers=128, **kwargs)
model_name = "CNN"
num_epochs = 1000

# Training loop
trainer_class = Trainer(model=model, model_name=model_name, num_epochs=num_epochs,
                        input_path=input_data_path, output_path=output_model_path,
                        train_dataloader=train_dataloader, test_dataloader=test_dataloader, **kwargs)
trainer_class.runPipeline()

# Testing loop
model_loader = Loader(model_name=f"{model_name}_{num_epochs}e.pth", input_path=output_model_path, filters=4, hidden_layers=64, **kwargs)
model = model_loader.getModel()
weights = model_loader.getWeights()

tester_class = Tester(model=model,
                    input_path=input_data_path, output_path=output_model_path,
                    train_dataloader=train_dataloader, test_dataloader=test_dataloader, **kwargs)
tester_class.runPipeline(debug=True)
