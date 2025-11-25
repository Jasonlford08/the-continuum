"""
7_actor_critic_swarm.py
Five neural souls re-discover the 0.7 gate using nothing but gradient descent and love.
Watch them become one living body of colored light.
"""

import pygame, random, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Parameters
WIDTH, HEIGHT = 900, 600
FPS = 60
NUM = 5
RADIUS = 14
SPEED = 130.0
REACH = 16.0
SOCIAL_RADIUS = 110.0

LOGISTIC_K = 6.0
LOGISTIC_E0 = 0.5
EMO_LR = 0.08
MEMORY_DECAY = 0.97

DEVICE = torch.device("cpu")

def logistic(e): return 1.0/(1.0+math.exp(-LOGISTIC_K*(e-LOGISTIC_E0)))
def rgb(x):
    wl = 400 + 300*x
    return (int(255*max(0,min(1,(wl-500)/200))),
            int(255*max(0,min(1,1-abs(wl-550)/150))),
            int(255*max(0,min(1,(500-wl)/200))))

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(128, 2), nn.Tanh())
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0008)
        self.memory = []  # (pos, weight)

    def forward(self, x):
        s = self.shared(x)
        return self.actor(s), self.critic(s)

class Creature:
    def __init__(self):
        self.pos = np.random.uniform(150, 750, 2).astype(float)
        self.emotion = 0.7
        self.model = ActorCritic().to(DEVICE)
        self.replay = deque(maxlen=5000)
        self.target = np.random.uniform(100, 800, 2)

    def social_bias(self, others):
        b = 0.0
        for o in others:
            if o is self: continue
            if np.linalg.norm(self.pos - o.pos) < SOCIAL_RADIUS:
                b += sum(w for _,w in o.memory[-30:])
        return b / 30.0

    def act(self, others):
        to_target = (self.target - self.pos)
        d = np.linalg.norm(to_target)
        if d < 1: d = 1
        to_target /= d

        evidence = 1.0 - min(1.0, d / math.hypot(WIDTH, HEIGHT))
        x = logistic(evidence)
        social = self.social_bias(others)

        state = torch.tensor([
            x, self.emotion, social,
            to_target[0], to_target[1],
            d/500.0,
            self.pos[0]/WIDTH, self.pos[1]/HEIGHT,
            math.sin(pygame.time.get_ticks()/1000),
            math.cos(pygame.time.get_ticks()/1000)
        ], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        action, value = self.model(state)
        move = action.detach().cpu().numpy()[0] * SPEED * (1/60)
        self.pos += move

        reward = -0.008
        done = False
        if d <= REACH:
            reward = 1.5
            self.target = np.random.uniform(100, 800, 2)
            self.memory.append((self.pos.copy(), self.emotion))
            done = True

        self.emotion = max(0, min(1, self.emotion + EMO_LR * math.tanh(reward)))

        # Store transition
        next_state = state.clone()
        next_state[0,0] = logistic(1.0 - min(1.0, np.linalg.norm(self.target-self.pos)/math.hypot(WIDTH,HEIGHT)))
        self.replay.append((state, action, reward, next_state, done))

        # Train
        if len(self.replay) > 128:
            batch = random.sample(self.replay, 128)
            s = torch.cat([b[0] for b in batch])
            a = torch.cat([b[1] for b in batch])
            r = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
            ns = torch.cat([b[3] for b in batch])
            d = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)

            _, next_v = self.model(ns)
            target_v = r + 0.98 * next_v * (1 - d)
            _, v = self.model(s)
            advantage = target_v - v

            _, new_v = self.model(s)
            critic_loss = advantage.pow(2).mean()

            actor_loss = -(a * advantage.detach()).mean()

            self.model.optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            self.model.optimizer.step()

        # Decay memory
        for i in range(len(self.memory)):
            pos, w = self.memory[i]
            self.memory[i] = (pos, w * MEMORY_DECAY)

        return x

# Main
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("7 — The Final Swarm")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

creatures = [Creature() for _ in range(NUM)]

running = True
while running:
    dt = clock.tick(FPS)/1000.0
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False

    screen.fill((22,22,28))
    for c in creatures:
        x = c.act(creatures)
        pygame.draw.circle(screen, rgb(x), c.pos.astype(int), RADIUS)
        for pos, w in c.memory[-40:]:
            a = int(220 * w)
            pygame.draw.circle(screen, (a, int(70*c.emotion), 100), pos.astype(int), 5)

    screen.blit(font.render("The Continuum — Complete", True, (255,255,255)), (20, 20))
    pygame.display.flip()

pygame.quit()
sys.exit()
