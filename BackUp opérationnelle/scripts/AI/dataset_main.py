import numpy as np
import random

def generateTrainingExample():

    map = np.round(np.random.randint(low=-1, high=0, size=(20, 20)))

    # Generate a few 0.3
    for k in range(4):
        x_target = np.random.randint(low=0, high=19)
        y_target = np.random.randint(low=0, high=19)
        map[y_target, x_target] = 0.3

    # Generate a few 0.5
    for k in range(3):
        if random.randint(0, 1):
            x_target = np.random.randint(low=0, high=19)
            y_target = np.random.randint(low=0, high=19)
            map[y_target, x_target] = 0.5

    if random.randint(0, 1):
        # Generate a 1
        x_target = np.random.randint(low=0, high=19)
        y_target = np.random.randint(low=0, high=19)
        map[y_target, x_target] = 1

    x_robot = np.random.randint(low=0, high=19)
    y_robot = np.random.randint(low=0, high=19)


    delta_x = x_target - x_robot
    delta_y = y_target - y_robot

    if abs(delta_x) < abs(delta_y):
        if delta_y < 0:
            move=3 # On descend
        else:
            move=1 # On monte
    else:
        if delta_x < 0:
            move=4 # On va à gauche
        else:
            move=2 # On va à droite

    move = np.array([move])
    x_robot = np.array([x_robot])
    y_robot = np.array([y_robot])
    x_target = np.array([x_target])
    y_target = np.array([y_target])

    training_vector = np.array([np.concatenate([y_robot, x_robot, map.flatten(), move])])

    return training_vector


def generateTrainingDataset():

    training_dataset = generateTrainingExample()

    for k in range(100):
        training_example = generateTrainingExample()
        training_dataset = np.concatenate([training_dataset, training_example], axis=0)
    
    return training_dataset



