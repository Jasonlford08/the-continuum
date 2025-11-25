"""
6_multi_agent_rl_swarm.py
Five neural networks learn the continuum from scratch using raw reward.
"""

import pygame, random, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Parameters
WIDTH, HEIGHT = 900, 600
FPS = 60
NUM = 5
RADIUS = 12
SPEED = 120.0
REACH = 14.0
SOCIAL_RADIUS = 100.0

LOGISTIC_K = 6.0
LOGISTIC_E0 = 0.5
EMO_LR = 0.08
BASE_THRESHOLD = 0.8
GAMMA = 0.22
MEMORY_DECAY = 0.97

DEVICE = torch.device("cpu")

def logistic(e): return 1.0/(1.0+math.exp(-LOGISTIC_K*(e-LOGISTIC_E0)))
def rgb(x):
    wl = 400 + 300*x
    return (int(255*max(0,min(1,(wl-500)/200))),
            int(255*max(0,min(1,1-abs(wl-550)/150))),
            int(255*max(0,min(1,(500-wl)/200))))

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Tanh()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.emotion = 0.7
        self.pos = np.random.uniform(100, 800, 2).astype(float)
        self.target = np.random.uniform(100, 800, 2).astype(float)
        self.memory = []

    def social_bias(Self, others):
        b = 0.0
        for o in others:
            if o is self: continue
            if np.linalg.norm(self.pos - o.pos) < SOCIAL_RADIUS:
                b += sum(w for _,w in o.memory[-20:])
        return b / 20.0

    def forward(self, state):
        return self.net(state)

    def act(self, dt, others):
        to_target = (self.target - self.pos)
        dist = np.linalg.norm(to_target)
        if dist < 1: dist = 1
        to_target /= dist

        evidence = 1.0 - min(1.0, dist / math.hypot(WIDTH, HEIGHT))
        x = logistic(evidence)
        social = self.social_bias(others)

        state = torch.tensor([
            x, self.emotion, social,
            to_target[0], to_target[1],
            dist/500.0, self.pos[0]/WIDTH, self.pos[1]/HEIGHT
        ], dtype=torch.float32)

        move = self(state).detach().numpy() * SPEED * dt
        self.pos += move

        reward = -0.01
        if dist <= REACH:
            reward = 1.0
            self.target = np.random.uniform(100, 800, 2)
            self.memory.append((self.pos.copy(), self.emotion))

        self.emotion = max(0,min(1, self.emotion + EMO_LR*math.tanh(reward)))

        # simple REINFORCE update
        log_prob = -torch.sum(self(state)**2)
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return x

# Main
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("6 — RL Swarm")
clock = pygame.time.Clock()

agents = [Agent() for _ in range(NUM)]

running = True
while running:
    dt = clock.tick(FPS)/1000.0
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False

    screen.fill((22,22,28))
    for a in agents:
        x = a.act(dt, agents)
        pygame.draw.circle(screen, rgb(x), a.pos.astype(int), RADIUS)
        for pos, w in a.memory[-30:]:
            a_val = int(200 * w)
            pygame.draw.circle(screen, (a_val, int(60*a.emotion), 80), pos.astype(int), 3)

    pygame.display.flip()

pygame.quit()
sys.exit()
