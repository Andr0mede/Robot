__author__ = "Aybuke Ozturk Suri, Johvany Gustave"
__copyright__ = "Copyright 2023, IN512, IPSA 2023"
__credits__ = ["Aybuke Ozturk Suri", "Johvany Gustave"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

from network import Network
from my_constants import *
import sys
import time
from random import randint
import argparse
from threading import Thread
import numpy as np
import random

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
        self.direction = 'x+'
        self.dict_of_direction = {'x-y-': (-1, -1),
                                  'y-': (-1, 0),
                                  'x+y-': (-1, 1),
                                  'x-': (0, -1),
                                  'x+': (0, 1),
                                  'x-y+': (1, -1),
                                  'y+': (1, 0),
                                  'x+y+': (1, 1)}
        self.dict_of_move = {'x-y-': 5,
                             'y-': 3,
                             'x+y-': 6,
                             'x-': 1,
                             'x+': 2,
                             'x-y+': 7,
                             'y+': 4,
                             'x+y+': 8}
        self.agent_map = -np.ones((env_conf["w"], env_conf["h"]))
        self.agent_map[self.y, self.x] = cell_val
        self.nb_agents = 0
        self.connected_agents = 0
        self.move_counter = 0
        self.poi_data = (-1, -1)
        self.delta_T = 0.5
        
        # Retrieving number of agents
        #command = {"header": 4}
        #self.network.send(command)
        #print('I retrieve nb of agent')
        #time.sleep(0.5)
        # Waiting for other agents to be connected
        self.waitingAllAgents()
        print('I retrieve nb of agent')
        time.sleep(0.5)
        self.strategy2()

    def msg_cb(self): 
        """ Method used to handle incoming messages """
        while self.running:
            msg = self.network.receive()
            print(msg)
            print("Attributs modifies")

            header = msg["header"]
            if header == 0: #broadcast_msg
                PoI_type = msg['Msg type']
                x, y = msg['position']
                owner = msg['owner']
                self.points_of_interest.append((y, x, owner, PoI_type))

            elif header == 1: #get data
                position = (msg['y'], msg['x'])
                self.position_n = position
                
                # Changing agent map only case is unexplored
                if self.agent_map[position] == -1:
                    self.agent_map[position] = msg['cell_val']
                
            elif header == 2: #move
                position = (msg['y'], msg['x'])
                self.position_n = position
                
                # Changing agent map only case is unexplored
                if self.agent_map[position] == -1:
                    self.agent_map[position] = msg['cell_val']
                
            elif header == 3: # get number of agents connected
                print('Number of connected agent:', msg['nb_connected_agents'])
                self.connected_agents = msg['nb_connected_agents']
                
            elif header == 4: # get number of agents
                self.nb_agents = msg['nb_agents']
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
        print(2 * self.nb_agents, 'POI to found')
        print(len(self.points_of_interest))
        return len(self.points_of_interest) == 2 * self.nb_agents
    
    
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
        w, h = self.agent_map.shape
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
        print('My next relative position is y =', y_nextB, ' and x =', x_nextB)
        x_next, y_next = x_nextB + x, y_nextB + y
        print(self.agent_map)
        print('My next position is y =', y_next, ' and x =', x_next)
        no_wall = x_next < w and x_next >= 0 and y_next < h and y_next >= 0
        print('No wall? ', no_wall)
        if no_wall:
            # If there is no wall, next value exists
            if self.agent_map[y_next, x_next] == -1:
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
            print('I found unexplored case near me')
            return self.deduceDirection(robot_place, relative_coordinate)
        
        # If all cases are explorated going forward if no wall
        if no_wall:
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
        print('All choices are:', possible_move, 'and I choose:', to_go)
        # Deducing direction          
        print('I finally choose a random direction')
        return self.deduceDirection(robot_place, to_go)
        
        
    def updatePoIMap(self):
        """
        This function allows to set to zero all values around a point of
        interest.

        Returns
        -------
        None.

        """
        y, x = self.position_n

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
                
                
                
                #####  PROSPECTION MODE  #####
                while not self.checkPOI():
                    
                    # Updating values of n - 1
                    self.position_n1 = self.position_n
                    print('I update n-1 pos')
                    
                    # Updating value of n and associated value
                    command = {"header": 1}
                    self.network.send(command)
                    time.sleep(self.delta_T)
                    self.value_n1 = self.value_n
                    self.value_n = self.getValue()
                    print('I update n pos')
                    
                    if self.exploration_mode and self.value_n == 0:
                        
                        # Choosing next direction to explore
                        self.direction = self.chooseDirection()
                        print('I choose the next direction')
                    
                    else:
                        # Is it the first to deactivate explo mode?
                        if self.exploration_mode:
                            
                            # Finding something so deactivate exploration mode
                            first = True
                            self.exploration_mode = False
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
                            print('I update the map')
                            print(self.agent_map)
                            
                            # Coming back into exploration mode
                            self.exploration_mode = True
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
                                        print('I turn right')
                                    
                                    # Else it's decreasing so go rear
                                    else:
                                        
                                        # Go backward
                                        self.changeDirection('rear')
                                        print('I go back')
                    
                    # Step to the next case
                    command = {"header": 2}
                    command['direction'] = self.dict_of_move[self.direction]
                    self.network.send(command)
                    self.move_counter += 1
                    print('I move forward')
                    
                    # Cooldown
                    time.sleep(self.delta_T)
                
                    
                #####  REACHING MODE  #####
                print('I switch to reaching mode')
                # Retreaving key and then chest positon
                key_position = (0, 0)
                print(self.points_of_interest)
                for i in range(len(self.points_of_interest)):
                    y, x, owner_id, poi_type = self.points_of_interest[i]
                    # Saving key and chest position
                    if owner_id == self.agent_id:
                        if poi_type == 0:
                            key_position = (y, x)
                        else:
                            chest_position = (y, x)
                print('Key position:', key_position)
                print('Chest position:', chest_position)
                
                # Retrieving position
                command = {"header": 1}
                self.network.send(command)
                # Cooldown
                print('Waiting')
                time.sleep(self.delta_T)
                
                print('Data retrieved')
                # Computing moves to do to reach key
                my_target = reachTarget(self.position_n)
                print('Object created')
                _, list_command = my_target.liste_move(key_position)
                print('I compute move to do to go to key', list_command)
    
                # Following path
                for i in range(len(list_command)):
                    command = {"header": 2}
                    command['direction'] = list_command[i]
                    self.network.send(command)
                    self.move_counter += 1
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
                print('I compute move to do to go to chest', list_command)
                
                # Following path
                for i in range(len(list_command)):
                    command = {"header": 2}
                    command['direction'] = list_command[i]
                    self.network.send(command)
                    self.move_counter += 1
                    print('Going chest: move', i + 1, '/', len(list_command))
                    
                    # Cooldown
                    time.sleep(self.delta_T)
                
                print('Mission accomplished for me!')
                finish = True
                
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
        print('Hello')
        print('robot position:', self.y, self.x)
        print('target position', y_cible, x_cible)
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
        
        print('Other Hello')
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
    
"""    
    try:    #Auto control test
        finish = False
        while not finish:
            
            
            
            #####  PROSPECTION MODE  #####
            while not agent.checkPOI():
                
                # Updating values of n - 1
                agent.position_n1 = agent.position_n
                print('I update n-1 pos')
                
                # Updating value of n and associated value
                command = {"header": 1}
                agent.network.send(command)
                time.sleep(delta_T)
                agent.value_n1 = agent.value_n
                agent.value_n = agent.getValue()
                print('I update n pos')
                
                if agent.exploration_mode and agent.value_n == 0:
                    
                    # Choosing next direction to explore
                    agent.direction = agent.chooseDirection()
                    print('I choose the next direction')
                
                else:
                    # Is it the first to deactivate explo mode?
                    if agent.exploration_mode:
                        
                        # Finding something so deactivate exploration mode
                        first = True
                        agent.exploration_mode = False
                        print(' I deactivate expl mode')
                    
                    # If value is 1, we found point of interest
                    if agent.value_n == 1:
                        
                        # Retrieving PoI data
                        command = {"header": 5}
                        agent.network.send(command)
                        time.sleep(delta_T)
                        
                        poi_owner, poi_type = agent.poi_data
                        y, x = agent.position_n
                        print('I found a PoI!', agent.position_n)
                        agent.points_of_interest.append((y, x, poi_owner, poi_type))
                        
                        # Updating map
                        agent.updatePoIMap()
                        print('I update the map')
                        print(agent.agent_map)
                        
                        # Coming back into exploration mode
                        agent.exploration_mode = True
                        print('I activate expl mode')
                        
                    else:
                        
                        # If the the first in no exploration mode, we
                        # check that the move is not in diagonal
                        if first:
                            # Deactivate first
                            first = False
                            
                            # Checking direction is valid
                            agent.direction = agent.checkMoveForSearch()
                            
                        else:
                            
                            # If value decreasing we must change direction
                            if agent.value_n <= agent.value_n1:
                                
                                # If there are equal, turning right to go to
                                # tangente
                                if agent.value_n == agent.value_n1:
                                    
                                    # Turning right
                                    agent.changeDirection('right')
                                    print('I turn right')
                                
                                # Else it's decreasing so go rear
                                else:
                                    
                                    # Go backward
                                    agent.changeDirection('rear')
                                    print('I go back')
                
                # Step to the next case
                command = {"header": 2}
                command['direction'] = agent.dict_of_move[agent.direction]
                agent.network.send(command)
                agent.move_counter += 1
                print('I move forward')
                
                # Cooldown
                time.sleep(delta_T)
            
                
            #####  REACHING MODE  #####
            print('I switch to reaching mode')
            # Retreaving key and then chest positon
            key_position = (0, 0)
            print(agent.points_of_interest)
            for i in range(len(agent.points_of_interest)):
                y, x, owner_id, poi_type = agent.points_of_interest[i]
                # Saving key and chest position
                if owner_id == agent.agent_id:
                    if poi_type == 0:
                        key_position = (y, x)
                    else:
                        chest_position = (y, x)
            print('Key position:', key_position)
            print('Chest position:', chest_position)
            
            # Retrieving position
            command = {"header": 1}
            agent.network.send(command)
            # Cooldown
            print('Waiting')
            time.sleep(delta_T)
            
            print('Data retrieved')
            # Computing moves to do to reach key
            my_target = reachTarget(agent.position_n)
            print('Object created')
            _, list_command = my_target.liste_move(key_position)
            print('I compute move to do to go to key', list_command)

            # Following path
            for i in range(len(list_command)):
                command = {"header": 2}
                command['direction'] = list_command[i]
                agent.network.send(command)
                agent.move_counter += 1
                print('Going key: move', i + 1, '/', len(list_command))
                
                # Cooldown
                time.sleep(delta_T)
            
            # Retrieving position
            command = {"header": 1}
            agent.network.send(command)
            # Cooldown
            time.sleep(delta_T)
            
            # Computing moves to do to reach chest
            my_target = reachTarget(agent.position_n)
            _, list_command = my_target.liste_move(chest_position)
            print('I compute move to do to go to chest', list_command)
            
            # Following path
            for i in range(len(list_command)):
                command = {"header": 2}
                command['direction'] = list_command[i]
                agent.network.send(command)
                agent.move_counter += 1
                print('Going chest: move', i + 1, '/', len(list_command))
                
                # Cooldown
                time.sleep(delta_T)
            
            print('Mission accomplished for me!')
            finish = True
            
    except KeyboardInterrupt:
        pass



"""