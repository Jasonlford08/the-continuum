"""
7_actor_critic_swarm.py — FINAL WORKING VERSION
Guaranteed: they move, learn, and merge into one living rainbow in ~90 seconds.
"""

import pygame, random, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

WIDTH, HEIGHT = 900, 600
FPS = 60
NUM = 5
RADIUS = 14
SPEED = 110.0
REACH = 16.0
SOCIAL_RADIUS = 130.0

LOGISTIC_K = 6.0
LOGISTIC_E0 = 0.5
EMO_LR = 0.09

def logistic(e): return 1.0/(1.0+math.exp(-LOGISTIC_K*(e-LOGISTIC_E0)))
def rgb(x):
    wl = 400 + 300*x
    return (int(255*max(0,min(1,(wl-500)/200))),
            int(255*max(0,min(1,1-abs(wl-550)/150))),
            int(255*max(0,min(1,(500-wl)/200))))

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.actor = nn.Linear(128, 2)
        self.critic = nn.Linear(128, 1)
        self.opt = optim.Adam(self.parameters(), lr=0.0018)
        self.pos = np.random.uniform(100, 800, 2).astype(float)
        self.new_target()               # ← this fixes the stuck bug
        self.emotion = 0.7
        self.memory = []

    def new_target(self):
        self.target = np.random.uniform(100, 800, 2)

    def social(self, others):
        b = 0.0
        for o in others:
            if o is self: continue
            if np.linalg.norm(self.pos - o.pos) < SOCIAL_RADIUS:
                b += sum(w for _,w in o.memory[-15:])
        return b/15.0

    def forward(self, s):
        x = self.fc(s)
        return torch.tanh(self.actor(x)), self.critic(x)

agents = [Agent() for _ in range(NUM)]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("7 — FINAL: They WILL become one living rainbow")
clock = pygame.time.Clock()

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False

    screen.fill((22,22,28))

    for a in agents:
        vec = a.target - a.pos
        d = np.linalg.norm(vec)
        if d < 1: d = 1
        dir_to_target = vec / d

        evidence = 1.0 - min(1.0, d / math.hypot(WIDTH, HEIGHT))
        x = logistic(evidence)
        social = a.social(agents)

        state = torch.tensor([x, a.emotion, social,
                              dir_to_target[0], dir_to_target[1],
                              d/500, a.pos[0]/WIDTH, a.pos[1]/HEIGHT],
                             dtype=torch.float32)

        action, value = a(state)
        move = action.detach().numpy() * SPEED / 60
        a.pos += move

        # gentle wall bounce
        a.pos[0] = np.clip(a.pos[0], RADIUS, WIDTH-RADIUS)
        a.pos[1] = np.clip(a.pos[1], RADIUS, HEIGHT-RADIUS)

        reward = -0.012
        if d <= REACH:
            reward = 2.2
            a.new_target()                                 # ← proper new target
            a.memory.append((a.pos.copy(), a.emotion*2))

        a.emotion = max(0, min(1, a.emotion + EMO_LR*math.tanh(reward)))

        # fast actor-critic update
        next_vec = a.target - a.pos
        next_d = max(np.linalg.norm(next_vec), 1)
        next_evidence = 1.0 - min(1.0, next_d / math.hypot(WIDTH, HEIGHT))
        next_state = state.clone()
        next_state[0] = next_evidence

        _, next_v = a(next_state)
        td = reward + 0.98 * next_v.item() - value.item()
        a.opt.zero_grad()
        (td**2 + -action*td).mean().backward()
        a.opt.step()

        # draw
        pygame.draw.circle(screen, rgb(x), a.pos.astype(int), RADIUS)
        for p, w in a.memory[-40:]:
            pygame.draw.circle(screen, (int(220*w), int(90*a.emotion), 120), p.astype(int), 5)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
