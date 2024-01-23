__author__ = "Aybuke Ozturk Suri, Johvany Gustave"
__copyright__ = "Copyright 2023, IN512, IPSA 2023"
__credits__ = ["Aybuke Ozturk Suri", "Johvany Gustave"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

import pygame, os
import time
from my_constants import *    

img_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "img")
safe_mode = False

class GUI:    
    def __init__(self, game, fps=10, cell_size=40):
        if not safe_mode:
            pygame.mixer.init()
            pygame.mixer.music.load(img_folder + "/wide_putin_music.ogg")
            pygame.mixer.music.set_volume(0.5)
            pygame.mixer.music.play(-1)
            
        self.frame1 = True
        self.game = game
        self.w, self.h = self.game.map_w, self.game.map_h
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.cell_size = cell_size
        self.screen_res = (self.w*cell_size, self.h*cell_size)
        self.player1_move = []
        self.player2_move = []
        self.player3_move = []
        self.player4_move = []
        self.move_to_draw = []
        self.agents = []


    def on_init(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_res)
        pygame.display.set_icon(pygame.image.load(img_folder + "/icon.png"))
        pygame.display.set_caption("IN512 Project")
        self.create_items()        
        self.running = True


    def create_items(self):
        #box
        box_img = pygame.image.load(img_folder + "/box.png")
        box_img = pygame.transform.scale(box_img, (self.cell_size, self.cell_size))
        self.boxes = [box_img.copy() for _ in range(self.game.nb_agents)]
        #keys
        key_img = pygame.image.load(img_folder + "/key.png")
        key_img = pygame.transform.scale(key_img, (self.cell_size, self.cell_size))
        self.keys = [key_img.copy() for _ in range(self.game.nb_agents)]
        #agent text number
        font = pygame.font.SysFont("Arial", self.cell_size//4, True)
        self.text_agents = [font.render(f"{i+1}", True, self.game.agents[i].color) for i in range(self.game.nb_agents)]
        
        if safe_mode:
            agent_img = pygame.image.load(img_folder + "/robot.png")
            agent_img = pygame.transform.scale(agent_img, (self.cell_size, self.cell_size))
            self.agents = [agent_img.copy() for _ in range(self.game.nb_agents)]
        else:
            # Player 1: Putin
            agent_img1_1 = pygame.image.load(img_folder + "/wide_putin1.png")
            agent_img1_2 = pygame.image.load(img_folder + "/wide_putin2.png")
            agent_img1_1 = pygame.transform.scale(agent_img1_1, (self.cell_size, self.cell_size))
            agent_img1_2 = pygame.transform.scale(agent_img1_2, (self.cell_size, self.cell_size))
            # Player 2: Modi
            agent_img2_1 = pygame.image.load(img_folder + "/wide_modi1.png")
            agent_img2_2 = pygame.image.load(img_folder + "/wide_modi2.png")
            agent_img2_1 = pygame.transform.scale(agent_img2_1, (self.cell_size, self.cell_size))
            agent_img2_2 = pygame.transform.scale(agent_img2_2, (self.cell_size, self.cell_size))
            # Player 3: Elizabeth
            agent_img3_1 = pygame.image.load(img_folder + "/wide_elizabeth1.png")
            agent_img3_2 = pygame.image.load(img_folder + "/wide_elizabeth2.png")
            agent_img3_1 = pygame.transform.scale(agent_img3_1, (self.cell_size, self.cell_size))
            agent_img3_2 = pygame.transform.scale(agent_img3_2, (self.cell_size, self.cell_size))
            # Player 4: Trump
            agent_img4_1 = pygame.image.load(img_folder + "/wide_trump1.png")
            agent_img4_2 = pygame.image.load(img_folder + "/wide_trump2.png")
            agent_img4_1 = pygame.transform.scale(agent_img4_1, (self.cell_size, self.cell_size))
            agent_img4_2 = pygame.transform.scale(agent_img4_2, (self.cell_size, self.cell_size))
            
            self.agents = [[agent_img1_1, agent_img1_2],
                           [agent_img2_1, agent_img2_2],
                           [agent_img3_1, agent_img3_2],
                           [agent_img4_1, agent_img4_2]]

    
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False

    
    def on_cleanup(self):
        pygame.event.pump()
        pygame.quit()
    

    def render(self):
        try:
            self.on_init()
            while self.running:
                for event in pygame.event.get():
                    self.on_event(event)    
                self.draw()
                self.clock.tick(self.fps)
            self.on_cleanup()
        except Exception:
            pass
    

    def draw(self):
        self.screen.fill(BG_COLOR)
        
        # Saving all robots positions only if they move
        if len(self.player1_move) == 0 or self.player1_move[-1] != (self.game.agents[0].x, self.game.agents[0].y):
            self.player1_move.append((self.game.agents[0].x, self.game.agents[0].y))
            self.move_to_draw.append((self.game.agents[0].x, self.game.agents[0].y, 0))
        if self.game.nb_agents > 1 and (len(self.player2_move) == 0 or self.player2_move[-1] != (self.game.agents[1].x, self.game.agents[1].y)):
            self.player2_move.append((self.game.agents[1].x, self.game.agents[1].y))
            self.move_to_draw.append((self.game.agents[1].x, self.game.agents[1].y, 1))
        if self.game.nb_agents > 2 and (len(self.player3_move) == 0 or self.player3_move[-1] != (self.game.agents[2].x, self.game.agents[2].y)):
            self.player3_move.append((self.game.agents[2].x, self.game.agents[2].y))
            self.move_to_draw.append((self.game.agents[2].x, self.game.agents[2].y, 2))
        if self.game.nb_agents > 3 and (len(self.player4_move) == 0 or self.player4_move[-1] != (self.game.agents[3].x, self.game.agents[3].y)):
            self.player4_move.append((self.game.agents[3].x, self.game.agents[3].y))
            self.move_to_draw.append((self.game.agents[3].x, self.game.agents[3].y, 3))
        # Robots path
        for i in range(len(self.move_to_draw)):
            x, y, robot_id = self.move_to_draw[i]
            pygame.draw.rect(self.screen, self.game.agents[robot_id].color, (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))

        #Grid
        for i in range(1, self.h):
            pygame.draw.line(self.screen, BLACK, (0, i*self.cell_size), (self.w*self.cell_size, i*self.cell_size))
        for j in range(1, self.w):
            pygame.draw.line(self.screen, BLACK, (j*self.cell_size, 0), (j*self.cell_size, self.h*self.cell_size))

        for i in range(self.game.nb_agents):            
            #keys
            pygame.draw.rect(self.screen, self.game.agents[i].color, (self.game.keys[i].x*self.cell_size, self.game.keys[i].y*self.cell_size, self.cell_size, self.cell_size), width=3)
            self.screen.blit(self.keys[i], self.keys[i].get_rect(topleft=(self.game.keys[i].x*self.cell_size, self.game.keys[i].y*self.cell_size)))
            
            #boxes
            pygame.draw.rect(self.screen, self.game.agents[i].color, (self.game.boxes[i].x*self.cell_size, self.game.boxes[i].y*self.cell_size, self.cell_size, self.cell_size), width=3)
            self.screen.blit(self.boxes[i], self.boxes[i].get_rect(topleft=(self.game.boxes[i].x*self.cell_size, self.game.boxes[i].y*self.cell_size)))
            
            #agents
            if safe_mode:
                self.screen.blit(self.agents[i], self.agents[i].get_rect(center=(self.game.agents[i].x*self.cell_size + self.cell_size//2, self.game.agents[i].y*self.cell_size + self.cell_size//2)))
            else:
                self.screen.blit(self.agents[i][int(self.frame1)], self.agents[i][int(self.frame1)].get_rect(center=(self.game.agents[i].x*self.cell_size + self.cell_size//2, self.game.agents[i].y*self.cell_size + self.cell_size//2)))
            
            self.screen.blit(self.text_agents[i], self.text_agents[i].get_rect(center=(self.game.agents[i].x*self.cell_size + self.cell_size-self.text_agents[i].get_width()//2, self.game.agents[i].y*self.cell_size + self.cell_size-self.text_agents[i].get_height()//2)))
        
        
        self.frame1 = not self.frame1
        time.sleep(0.1)
        pygame.display.update()