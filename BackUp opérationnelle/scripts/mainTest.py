from game import Game
from agent import Agent
from my_constants import *
import matplotlib.pyplot as plt
import time

# game_instance = Game(nb_agents=2, map_id=1)
# agent_instance = Agent()
# map_real = game_instance.map_real
# plt.imshow(map_real)
# plt.title("map real")
# plt.show()



if __name__ == "__main__":
    from random import randint
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--server_ip", help="Ip address of the server", type=str, default="localhost")
    args = parser.parse_args()

    agent = Agent(args.server_ip)
    try:    #Auto control test
        while True:
            cmds = {"header": agent.take_decision()}
            if cmds["header"] == BROADCAST_MSG:
                cmds["Msg type"] = int(input("1 <-> Key discovered\n2 <-> Box discovered\n3 <-> Completed\n"))
                cmds["position"] = (agent.x, agent.y)
                cmds["owner"] = agent.agent_id-1 # TODO: specify the owner of the item
            elif cmds["header"] == MOVE:
                cmds["direction"] = agent.move_decision()
            agent.network.send(cmds)

            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
