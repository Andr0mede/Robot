__author__ = "Aybuke Ozturk Suri, Johvany Gustave"
__copyright__ = "Copyright 2023, IN512, IPSA 2023"
__credits__ = ["Aybuke Ozturk Suri", "Johvany Gustave"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"



#                          _
#    ___  ___  ___  _ __  | |     ___
#   / _ |/ _ |/ _ \| /_ \ | _\   |
#  | (_|| (_||  __/| / \ || |  _ |
#   \___|\__ |\__\ |_| |_|\__\|_||
#        /__/

from network import Network
from my_constants import *
import sys
import time
from random import randint
import argparse
from threading import Thread
import numpy as np
import random
import os
import torch
from AI.nn_main import decideMove

class Agent:
    """ Class that implements the behaviour of each agent based on their perception and communication with other agents """
    def __init__(self, server_ip):
        
 
        #DO NOT TOUCH THE FOLLOWING INSTRUCTIONS
        self.network = Network(server_ip=server_ip)
        self.agent_id = self.network.id
        self.running = True
        self.network.send({"header": GET_DATA})
        env_conf = self.network.receive()
        self.x, self.y = env_conf["x"], env_conf["y"]   #initial agent position
        self.w, self.h = env_conf["w"], env_conf["h"]   #environment dimensions
        print('Env conf:', self.w, self.h)
        cell_val = env_conf["cell_val"] #value of the cell the agent is located in
        Thread(target=self.msg_cb, daemon=True).start()
        
        #TODO: DEINE YOUR ATTRIBUTES HERE
        self.exploration_mode = True
        self.points_of_interest = [] # tuple with y, x, owner id, and type
        self.position_n = (self.y, self.x)
        self.position_n1 = (self.y, self.x)
        self.value_n = 0
        self.value_n1 = 0
        
        #   x-y- | y- | x+y-       x
        #   -----+----+-----     +----->
        #   x-   | O  | x+     y |
        #   -----+----+-----     |
        #   x-y+ |y+  | x+y+     v
        self.dict_of_direction = {'x-y-': (-1, -1),
                                  'y-': (-1, 0),
                                  'x+y-': (-1, 1),
                                  'x-': (0, -1),
                                  'x+': (0, 1),
                                  'x-y+': (1, -1),
                                  'y+': (1, 0),
                                  'x+y+': (1, 1)}
        self.direction = random.choice(list(self.dict_of_direction.items()))[0]
        self.dict_of_move = {'x-y-': 5,
                             'y-': 3,
                             'x+y-': 6,
                             'x-': 1,
                             'x+': 2,
                             'x-y+': 7,
                             'y+': 4,
                             'x+y+': 8}
        self.agent_map = -np.ones((self.h, self.w))
        self.agent_map[self.y, self.x] = cell_val
        self.nb_agents = 0
        self.connected_agents = 0
        self.move_counter = 0
        self.poi_data = (-1, -1)
        self.delta_T = 0.1
        self.output_model_path = os.path.abspath("")
        self.debug_mode = False
        
        # Waiting for other agents to be connected
        self.waitingAllAgents()
        if self.debug_mode:
            print('I retrieve nb of agent')
        time.sleep(0.5)
        # Choosing strategy
        self.strategy3()

    def msg_cb(self): 
        """ Method used to handle incoming messages """
        while self.running:
            msg = self.network.receive()
            if self.debug_mode:
                print(msg)
                print("Attributs modifies")

            header = msg["header"]
            if header == 0: #broadcast_msg
                PoI_type = msg['Msg type'] - 1
                x, y = msg['position']
                owner = msg['owner']
                self.points_of_interest.append((y, x, owner, PoI_type))
                self.updatePoIMap()
                self.exploration_mode = True

            elif header == 1 or header == 2: #get data or move
                position = (msg['y'], msg['x'])
                self.position_n = position
                
                # Changing agent map only case is unexplored
                if self.agent_map[position] == -1:
                    self.agent_map[position] = msg['cell_val']
                
            elif header == 3: # get number of agents connected
                if self.debug_mode:
                    print('Number of connected agent:', msg['nb_connected_agents'])
                self.connected_agents = msg['nb_connected_agents']
                
            elif header == 4: # get number of agents
                self.nb_agents = msg['nb_agents']
                if self.debug_mode:
                    print('Number of agents is: ', msg['nb_agents'])
            
            elif header == 5: # get item owner
                poi_owner = msg['owner']
                poi_type = msg['type']
                self.poi_data = (poi_owner, poi_type)
            

    def checkPOI(self):
        """
        This function allows to check if the Point of Interest list is
        completed or not. It takes no input and return a boolean if yes or not
        the list is complete

        Returns
        -------
        TYPE: Boolean
            DESCRIPTION: If yes or not the list of PoI is completed

        """
        if self.debug_mode:
            print(2 * self.nb_agents, 'POI to found')
        # Deleting potential tuple in double
        seen = set()
        result = []
        for tuple_elem in self.points_of_interest:
            # Converting tuple into frozenset to make it hashable
            frozen_tuple = frozenset(tuple_elem)
            # Checking if it already exists
            if frozen_tuple not in seen:
                seen.add(frozen_tuple)
                result.append(tuple_elem)
        
        if self.debug_mode:
            print(len(result))
        return len(result) == 2 * self.nb_agents
    
    
    def getValue(self):
        """
        This function allows to get the value on the agent map

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.agent_map[self.position_n]
    
    
    def deduceDirection(self, initial_position, final_position):
        """
        This function allows to deduce the direction to follow to go from 
        initial position to final position. It takes as inputs initial and
        final postion in tuple. It returns a string containing the direction

        Parameters
        ----------
        initial_position : tuple (y, x)
            DESCRIPTION: Initial position
        final_position : tuple (y, x)
            DESCRIPTION: Final position

        Returns
        -------
        TYPE: String
            DESCRIPTION: Direction to follow

        """
        # Retrieving values and computing delta
        yf, xf = final_position
        yi, xi = initial_position
        # Normalization
        difference = (np.sign(yf - yi), np.sign(xf - xi))
        return self.retrieveDirection(self.dict_of_direction, difference)
    
    
    def retrieveDirection(self, my_dict, value):
        """
        This function allows to found the key associated to a value in a
        dictionnary. It takes as inputs the dict and the value and returns the
        key associated

        Parameters
        ----------
        my_dict : dictionnary
            DESCRIPTION: Dict where to find the key
        value : tuple
            DESCRIPTION: Value to find the key

        Returns
        -------
        TYPE: String
            DESCRIPTION: Key corresponding to the value

        """
        return list(my_dict.keys())[list(my_dict.values()).index(value)]
    
    
    def chooseDirection(self):
        """
        This function allows to the agent to choose the direction to take after
        a moove in exploration mode. It takes no argument and return a string
        containing direction

        Returns
        -------
        TYPE: String
            DESCRIPTION: Direction to take

        """
        h, w = self.agent_map.shape
        y, x = self.position_n
        
        # Finding the size of the under-matrix of environment
        if x == 0 or x == w - 1:
            if y == 0 or y == h - 1:
                around = (2, 2)  # Corner: size 2*2
            else:
                around = (3, 2)  # Vertical edge: size 3*2
        elif y == 0 or y == h - 1:
            around = (2, 3)  # Horizontal edge: size 2*3
        else:
            around = (3, 3)  # Center: size 3*3
        # Deducing robot position in the under-matrix  
        robot_place = (int(y != 0), int(x != 0))
        if self.debug_mode:
            print('the size of UM is ', around, ' and Im in ', robot_place)
        
        # Computing indexes of extraction
        x_start, y_start = max(0, x - around[1] // 2), max(0, y - around[0] // 2)
        x_end, y_end = min(w, x_start + around[1]), min(h, y_start + around[0])
        
        # Extracting under-array of the map
        B = np.copy(self.agent_map[y_start:y_end, x_start:x_end])
        B[robot_place] = -666
        
        # First we check if the value in front of the robot exists and it's
        # unexplored
        y_nextB, x_nextB = self.dict_of_direction[self.direction]
        if self.debug_mode:
            print('My next relative position is y =', y_nextB, ' and x =', x_nextB)
        x_next, y_next = x_nextB + x, y_nextB + y
        if self.debug_mode:
            print(self.agent_map)
            print('My next position is y =', y_next, ' and x =', x_next)
        no_wall = x_next < w and x_next >= 0 and y_next < h and y_next >= 0
        if self.debug_mode:
            print('No wall? ', no_wall)
        if no_wall:
            # If there is no wall, next value exists
            if self.agent_map[y_next, x_next] == -1:
                if self.debug_mode:
                    print('Case in front is unexplored')
                # If yes, we continue in the same direction
                return self.direction
        
        # If not go one the first unexplored case
        i_max, j_max = around
        found = False
        i = 0
        while i < i_max and not found:
            j = 0
            while j < j_max and not found:
                if B[i, j] == -1:
                    relative_coordinate = (i, j)
                    found = True
                j += 1
            i += 1
        
        # If -1 found, there is unexplored case so we go on it
        if found:
            if self.debug_mode:
                print('I found unexplored case near me')
            return self.deduceDirection(robot_place, relative_coordinate)
        
        # If all cases are explorated going forward if no wall
        if no_wall:
            if self.debug_mode:
                print('All cases are explorated near me')
            return self.direction
        
        # If there is a wall choosing randomly a direction
        # Saving all possible moves
        possible_move =[]
        for i in range(i_max):
            for j in range(j_max):
                if B[i, j] == 0:
                    possible_move.append((i, j))
        # Choosing randomly a move
        to_go = random.choices(possible_move)[0]
        if self.debug_mode:
            print('All choices are:', possible_move, 'and I choose:', to_go)
            # Deducing direction          
            print('I finally choose a random direction')
        return self.deduceDirection(robot_place, to_go)
    
    
    def chooseDirectionAI(self, agent_map, x_robot, y_robot):
        """
        This function allows to the agent to choose the direction with the AI 
        to take after a moove. It takes no argument and return a string
        containing direction

        Returns
        -------
        TYPE: String
            DESCRIPTION: Direction to take

        """
        x_robot = np.array([x_robot])
        y_robot = np.array([y_robot])
        eval_tensor = np.concatenate((y_robot, x_robot, agent_map.flatten()))
        eval_tensor = torch.from_numpy(eval_tensor).float().unsqueeze(0)
        in_channels=len(eval_tensor)
        output_size=5

        #try:
        model = LayeredNN(input_size=in_channels, output_size=output_size)
        model.load_state_dict(torch.load(f"{self.output_model_path}/LayeredNN_1000e.pth"))
            # model_loaded = "Currently trained model used "
        #except:
            #raise ImportError("Can't find/load/fit NN weights")

        model.eval()
        with torch.no_grad():

            prediction = torch.argmax(model(eval_tensor), dim=1).numpy()

        return prediction    
    
        
    def updatePoIMap(self):
        """
        This function allows to set to zero all values around a point of
        interest.

        Returns
        -------
        None.

        """
        # Updating map with all poi already found
        for i in range(len(self.points_of_interest)):
            # Retrieving poi coordinate
            poi_data = self.points_of_interest[i]
            y, x, _, _ = poi_data
            
            # Set to 0 only if it is inside the map
            for i in range(max(0, x - 2), min(self.w, x + 3)):
                for j in range(max(0, y - 2), min(self.h, y + 3)):
                    self.agent_map[j][i] = 0
        
        return


    def changeDirection(self, back_or_right):
        """
        This function allows to the robot to turn right or backin function of 
        its own position.

        Returns
        -------
        None.

        """
        # All direction in order
        direction_list = ['y-', 'x+y-', 'x+', 'x+y+', 'y+', 'x-y+', 'x-', 'x-y-']
        size = len(direction_list)
        # Retrieving current direction
        current_direction = self.direction
        # Finding the index
        index = direction_list.index(current_direction)
        # Turning back or right ?
        offset = 2 if (back_or_right == 'right') else 4
        # Updating direction
        self.direction = direction_list[(index + offset) % size]
        return
    
    
    def checkMoveForSearch(self):
        """
        This function allows to check that the move is horizontal or vertical.
        If not, it changes it to a valid move. It takes as input nothing and
        return the move to do

        Returns
        -------
        TYPE: str
            DESCRIPTION: A valid move to do

        """
        # Retrieving direction
        actual_move = self.direction
        
        # If move is already valid, we do not change it
        if len(actual_move) == 2:
            return actual_move
        
        # If not, we do a projection
        projection_dict = {'x-y-': 'x-',
                           'x+y-': 'y-',
                           'x-y+': 'y+',
                           'x+y+': 'x+'}
        return projection_dict[actual_move]
    
    
    def waitingAllAgents(self):
        """
        Function allowing the agent to wait for other agents to be connected

        Returns
        -------
        None.

        """
        command = {"header": 4}
        self.network.send(command)
        time.sleep(0.5)
        check = True
        while check:
            command = {"header": 3}
            self.network.send(command)
            time.sleep(0.5)
            
            print('Still waiting', self.connected_agents, self.nb_agents)
            # Retrieving number of connected agents
            if self.connected_agents == self.nb_agents:
                check = False

    def strategy1(self):
        while True:
            direction = randint(0, 9)
            cmd = {'header': 2, 'direction': direction}
            self.network.send(cmd)
            time.sleep(1)
            
            
    def strategy2(self):
        try:    #Auto control test
            finish = False
            while not finish:

                #####  FIRST PHASE  #####
                while not self.checkPOI():
                    
                    # Updating values of n - 1
                    self.position_n1 = self.position_n
                    if self.debug_mode:
                        print('I update n-1 pos')
                    
                    # Updating value of n and associated value
                    command = {"header": 1}
                    self.network.send(command)
                    time.sleep(self.delta_T)
                    self.value_n1 = self.value_n
                    self.value_n = self.getValue()
                    if self.debug_mode:
                        print('I update n pos')
                    
                    if self.exploration_mode and self.value_n == 0:
                    #####  EXPLORATION MODE  #####    
                        # Choosing next direction to explore
                        self.direction = self.chooseDirection()
                        if self.debug_mode:
                            print('I choose the next direction')
                    
                    else:
                    #####  PROSPECTION MODE  #####s
                        # Is it the first to deactivate explo mode?
                        if self.exploration_mode:
                            
                            # Finding something so deactivate exploration mode
                            first = True
                            self.exploration_mode = False
                            if self.debug_mode:
                                print(' I deactivate expl mode')
                        
                        # If value is 1, we found point of interest
                        if self.value_n == 1:
                            
                            # Retrieving PoI data
                            command = {"header": 5}
                            self.network.send(command)
                            time.sleep(self.delta_T)
                            
                            poi_owner, poi_type = self.poi_data
                            y, x = self.position_n
                            print('I found a PoI!', self.position_n)
                            self.points_of_interest.append((y, x, poi_owner, poi_type))
                            
                            # Updating map
                            self.updatePoIMap()
                            if self.debug_mode:
                                print('I update the map')
                                print(self.agent_map)
                            
                            # Communicate position to teammate
                            command = {'header': 0}
                            command['Msg type'] = poi_type + 1
                            command['position'] = (x, y)
                            command['owner'] = poi_owner
                            self.network.send(command)
                            time.sleep(self.delta_T)
                            
                            # Coming back into exploration mode
                            self.exploration_mode = True
                            if self.debug_mode:
                                print('I activate expl mode')
                            
                        else:
                            
                            # If the the first in no exploration mode, we
                            # check that the move is not in diagonal
                            if first:
                                # Deactivate first
                                first = False
                                
                                # Checking direction is valid
                                self.direction = self.checkMoveForSearch()
                                
                            else:
                                
                                # If value decreasing we must change direction
                                if self.value_n <= self.value_n1:
                                    
                                    # If there are equal, turning right to go to
                                    # tangente
                                    if self.value_n == self.value_n1:
                                        
                                        # Turning right
                                        self.changeDirection('right')
                                        if self.debug_mode:
                                            print('I turn right')
                                    
                                    # Else it's decreasing so go rear
                                    else:
                                        
                                        # Go backward
                                        self.changeDirection('rear')
                                        if self.debug_mode:
                                            print('I go back')
                    
                    # Step to the next case
                    command = {"header": 2}
                    command['direction'] = self.dict_of_move[self.direction]
                    self.network.send(command)
                    self.move_counter += 1
                    if self.debug_mode:
                        print('I move forward')
                    
                    # Cooldown
                    time.sleep(self.delta_T)
                
                #####  SECOND PHASE  #####    
                #####  REACHING MODE  #####
                print('I switch to reaching mode')
                # Retreaving key and then chest positon
                key_position = (0, 0)
                if self.debug_mode:
                    print(self.points_of_interest)
                for i in range(len(self.points_of_interest)):
                    y, x, owner_id, poi_type = self.points_of_interest[i]
                    # Saving key and chest position
                    if owner_id == self.agent_id:
                        if poi_type == 0:
                            key_position = (y, x)
                        else:
                            chest_position = (y, x)
                if self.debug_mode:
                    print('Key position:', key_position)
                    print('Chest position:', chest_position)
                
                # Retrieving position
                command = {"header": 1}
                self.network.send(command)
                # Cooldown
                if self.debug_mode:
                    print('Waiting')
                time.sleep(self.delta_T)
                
                if self.debug_mode:
                    print('Data retrieved')
                # Computing moves to do to reach key
                my_target = reachTarget(self.position_n)
                if self.debug_mode:
                    print('Object created')
                _, list_command = my_target.liste_move(key_position)
                if self.debug_mode:
                    print('I compute move to do to go to key', list_command)
    
                # Following path
                for i in range(len(list_command)):
                    command = {"header": 2}
                    command['direction'] = list_command[i]
                    self.network.send(command)
                    self.move_counter += 1
                    if self.debug_mode:
                        print('Going key: move', i + 1, '/', len(list_command))
                    
                    # Cooldown
                    time.sleep(self.delta_T)
                
                # Retrieving position
                command = {"header": 1}
                self.network.send(command)
                # Cooldown
                time.sleep(self.delta_T)
                
                # Computing moves to do to reach chest
                my_target = reachTarget(self.position_n)
                _, list_command = my_target.liste_move(chest_position)
                if self.debug_mode:
                    print('I compute move to do to go to chest', list_command)
                
                # Following path
                for i in range(len(list_command)):
                    command = {"header": 2}
                    command['direction'] = list_command[i]
                    self.network.send(command)
                    self.move_counter += 1
                    if self.debug_mode:
                        print('Going chest: move', i + 1, '/', len(list_command))
                    
                    # Cooldown
                    time.sleep(self.delta_T)
                
                print('Mission accomplished for me!')
                print('I have walked over:', self.move_counter, 'cases')
                finish = True
                
        except KeyboardInterrupt:
            pass


    def strategy3(self):
        try:    #Auto control test

            #####  FIRST PHASE  #####
            while not self.checkPOI():
                
                # Updating values of n - 1
                self.position_n1 = self.position_n
                if self.debug_mode:
                    print('I update n-1 pos')
                
                # Updating value of n and associated value
                command = {"header": 1}
                self.network.send(command)
                time.sleep(self.delta_T)
                self.value_n1 = self.value_n
                self.value_n = self.getValue()
                if self.debug_mode:
                    print('I update n pos')
                
                if self.value_n == 1:
  
                    # Retrieving PoI data
                    command = {"header": 5}
                    self.network.send(command)
                    time.sleep(self.delta_T)
                    
                    poi_owner, poi_type = self.poi_data
                    y, x = self.position_n
                    print('I found a PoI!', self.position_n)
                    self.points_of_interest.append((y, x, poi_owner, poi_type))
                    
                    # Updating map
                    self.updatePoIMap()
                    if self.debug_mode:
                        print('I update the map')
                        print(self.agent_map)
                    
                    # Communicate position to teammate
                    command = {'header': 0}
                    command['Msg type'] = poi_type + 1
                    command['position'] = (x, y)
                    command['owner'] = poi_owner
                    self.network.send(command)
                    time.sleep(self.delta_T)    

                # Choosing next direction to explore
                next_direction = decideMove(self.agent_map, self.position_n[1], self.position_n[0])
                next_direction += 1
                if next_direction == 1:
                    next_direction = 3
                elif next_direction == 3:
                    next_direction = 4
                elif next_direction == 4:
                    next_direction = 1
                
                self.direction = next_direction
                if self.debug_mode:
                    print('I choose the next direction')
                
                # Step to the next case
                command = {"header": 2}
                command['direction'] = self.direction
                self.network.send(command)
                self.move_counter += 1
                if self.debug_mode:
                    print('I move forward')
                
                # Cooldown
                time.sleep(self.delta_T)
            
            #####  SECOND PHASE  #####    
            #####  REACHING MODE  #####
            print('I switch to reaching mode')
            # Retreaving key and then chest positon
            key_position = (0, 0)
            if self.debug_mode:
                print(self.points_of_interest)
            for i in range(len(self.points_of_interest)):
                y, x, owner_id, poi_type = self.points_of_interest[i]
                # Saving key and chest position
                if owner_id == self.agent_id:
                    if poi_type == 0:
                        key_position = (y, x)
                    else:
                        chest_position = (y, x)
            if self.debug_mode:
                print('Key position:', key_position)
                print('Chest position:', chest_position)
            
            # Retrieving position
            command = {"header": 1}
            self.network.send(command)
            # Cooldown
            if self.debug_mode:
                print('Waiting')
            time.sleep(self.delta_T)
            
            if self.debug_mode:
                print('Data retrieved')
            # Computing moves to do to reach key
            my_target = reachTarget(self.position_n)
            if self.debug_mode:
                print('Object created')
            _, list_command = my_target.liste_move(key_position)
            if self.debug_mode:
                print('I compute move to do to go to key', list_command)

            # Following path
            for i in range(len(list_command)):
                command = {"header": 2}
                command['direction'] = list_command[i]
                self.network.send(command)
                self.move_counter += 1
                if self.debug_mode:
                    print('Going key: move', i + 1, '/', len(list_command))
                
                # Cooldown
                time.sleep(self.delta_T)
            
            # Retrieving position
            command = {"header": 1}
            self.network.send(command)
            # Cooldown
            time.sleep(self.delta_T)
            
            # Computing moves to do to reach chest
            my_target = reachTarget(self.position_n)
            _, list_command = my_target.liste_move(chest_position)
            if self.debug_mode:
                print('I compute move to do to go to chest', list_command)
            
            # Following path
            for i in range(len(list_command)):
                command = {"header": 2}
                command['direction'] = list_command[i]
                self.network.send(command)
                self.move_counter += 1
                if self.debug_mode:
                    print('Going chest: move', i + 1, '/', len(list_command))
                
                # Cooldown
                time.sleep(self.delta_T)
            
            print('Mission accomplished for me!')
            print('I have walked over:', self.move_counter, 'cases')

                
        except KeyboardInterrupt:
            pass



class reachTarget:
    def __init__(self, position):
        self.y = position[0]
        self.x = position[1]
        self.chemin = [(self.x, self.y)]
        self.directions = []
        
    def liste_move(self, cible):
        y_cible = cible[0]
        x_cible = cible[1]
        if (x_cible >= self.x) and (y_cible <= self.y): # en haut a droite
            while((self.chemin[-1][0] != x_cible) and (self.chemin[-1][1] != y_cible)):
                self.chemin.append((self.chemin[-1][0] + 1, self.chemin[-1][1] - 1))
            if (x_cible == self.chemin[-1][0]):
                while (y_cible != self.chemin[-1][1]):
                    self.chemin.append((self.chemin[-1][0], self.chemin[-1][1] - 1))
            if (y_cible == self.chemin[-1][1]):
                while (x_cible != self.chemin[-1][0]):
                    self.chemin.append((self.chemin[-1][0] + 1, self.chemin[-1][1]))
        
        if (x_cible <= self.x) and (y_cible <= self.y): # en haut a gauche
            while((self.chemin[-1][0] != x_cible) and (self.chemin[-1][1] != y_cible)):
                self.chemin.append((self.chemin[-1][0] - 1, self.chemin[-1][1] - 1))
            if (x_cible == self.chemin[-1][0]):
                while (y_cible != self.chemin[-1][1]):
                    self.chemin.append((self.chemin[-1][0], self.chemin[-1][1] - 1))
            if (y_cible == self.chemin[-1][1]):
                while (x_cible != self.chemin[-1][0]):
                    self.chemin.append((self.chemin[-1][0] - 1, self.chemin[-1][1]))
        
        if (x_cible <= self.x) and (y_cible >= self.y): # en bas a gauche
            while((self.chemin[-1][0] != x_cible) and (self.chemin[-1][1] != y_cible)):
                self.chemin.append((self.chemin[-1][0] - 1, self.chemin[-1][1] + 1))
            if (x_cible == self.chemin[-1][0]):
                while (y_cible != self.chemin[-1][1]):
                    self.chemin.append((self.chemin[-1][0], self.chemin[-1][1] + 1))
            if (y_cible == self.chemin[-1][1]):
                while (x_cible != self.chemin[-1][0]):
                    self.chemin.append((self.chemin[-1][0] - 1, self.chemin[-1][1]))
        
        if (x_cible >= self.x) and (y_cible >= self.y): # en bas a droite
            while((self.chemin[-1][0] != x_cible) and (self.chemin[-1][1] != y_cible)):
                self.chemin.append((self.chemin[-1][0] + 1, self.chemin[-1][1] + 1))
            if (x_cible == self.chemin[-1][0]):
                while (y_cible != self.chemin[-1][1]):
                    self.chemin.append((self.chemin[-1][0], self.chemin[-1][1] + 1))
            if (y_cible == self.chemin[-1][1]):
                while (x_cible != self.chemin[-1][0]):
                    self.chemin.append((self.chemin[-1][0] + 1, self.chemin[-1][1]))
        
        #for i in range(len(self.chemin)):
            #self.chemin[i] = (self.chemin[i][1], self.chemin[i][0])
        
        for i in range(1, len(self.chemin)):
            trajectoire = (self.chemin[i][0] - self.chemin[i-1][0], self.chemin[i][1] - self.chemin[i-1][1])
            if trajectoire == (-1, -1):
                self.directions.append(5)
            elif trajectoire == (0, -1):
                self.directions.append(3)
            elif trajectoire == (1, -1):
                self.directions.append(6)
            elif trajectoire == (-1, 0):
                self.directions.append(1)
            elif trajectoire == (1, 0):
                self.directions.append(2)
            elif trajectoire == (-1, 1):
                self.directions.append(7)
            elif trajectoire == (0, 1):
                self.directions.append(4)
            elif trajectoire == (1, 1):
                self.directions.append(8)
            
        return self.chemin, self.directions
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--server_ip", help="Ip address of the server", type=str, default="localhost")
    args = parser.parse_args()

    agent = Agent(args.server_ip)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--server_ip", help="Ip address of the server", type=str, default="localhost")
    args = parser.parse_args()

    agent = Agent(args.server_ip)
    delta_T = 0.5
    print('Lets start')
    
