import numpy as np
import random

def generateTrainingExample(flatten=True):

    if np.random.randint(0, 2):
        map = np.round(np.random.randint(low=-1, high=1, size=(20, 20)))
    else:
        map = np.round(np.ones((20, 20))*(-1))

    # Generate a few 0.3f
    for k in range(4):
        x_target = np.random.randint(low=0, high=20)
        y_target = np.random.randint(low=0, high=20)
        map[y_target, x_target] = 0.3

    # Generate a few 0.5
    for k in range(3):
        if np.random.randint(0, 2):
            x_target = np.random.randint(low=0, high=20)
            y_target = np.random.randint(low=0, high=20)
            map[y_target, x_target] = 0.5

    if np.random.randint(0, 2):
        # Generate a 1
        x_target = np.random.randint(low=0, high=20)
        y_target = np.random.randint(low=0, high=20)
        map[y_target, x_target] = 1

    x_robot = np.random.randint(low=0, high=20)
    y_robot = np.random.randint(low=0, high=20)


    delta_x = x_target - x_robot
    delta_y = y_target - y_robot

    if abs(delta_x) < abs(delta_y):
        if delta_y < 0:
            move=2 # On descend
        else:
            move=0 # On monte
    else:
        if delta_x < 0:
            move=3 # On va à gauche
        else:
            move=1 # On va à droite
    
    move = np.array([move])
    x_robot = np.array([x_robot])
    y_robot = np.array([y_robot])
    x_target = np.array([x_target])
    y_target = np.array([y_target])

    if flatten:
        return np.array([np.concatenate([y_robot, x_robot, map.flatten(), move])])
    else:
        array = np.pad(map, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        array[x_robot, 0] = 1
        array[x_robot, -1] = 1
        array[y_robot, 0] = 1
        array[y_robot, -1] = 1
        return np.expand_dims(array, axis=0), move 

    return None

def generateTrainingDataset(batch_size=100, flatten=True):

    if flatten:
        training_dataset = generateTrainingExample(flatten=flatten)
        for k in range(batch_size-1):
            training_example = generateTrainingExample(flatten=flatten)
            training_dataset = np.concatenate([training_dataset, training_example], axis=0)
        return training_dataset
    else:
        training_dataset, training_labels = generateTrainingExample(flatten=flatten)
        for k in range(batch_size-1):
            training_example, training_example_labels = generateTrainingExample(flatten=flatten)
            training_dataset = np.concatenate([training_dataset, training_example], axis=0)
            training_labels = np.concatenate([training_labels, training_example_labels], axis=0)
        return training_dataset, training_labels
    
    return None



