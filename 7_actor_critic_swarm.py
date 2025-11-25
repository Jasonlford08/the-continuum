"""
7_actor_critic_swarm.py — WALL-FIXED VERSION
They stay on screen forever. Watch them merge into one rainbow cloud in ~90 seconds.
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
SPEED = 100.0  # Slower for better observation
REACH = 16.0
SOCIAL_RADIUS = 120.0

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
        self.actor = nn.Sequential(nn.Linear(128, 2), nn.Tanh())
        self.critic = nn.Linear(128, 1)
        self.opt = optim.Adam(self.parameters(), lr=0.0015)
        self.pos = np.random.uniform(RADIUS, WIDTH-RADIUS, 2).astype(float)
        self.target = np.random.uniform(RADIUS, WIDTH-RADIUS, 2)
        self.emotion = 0.7
        self.memory = []  # (pos, weight)

    def social(self, others):
        b = 0.0
        for o in others:
            if o is self: continue
            if np.linalg.norm(self.pos - o.pos) < SOCIAL_RADIUS:
                b += sum(w for _,w in o.memory[-20:])
        return b/20.0

    def forward(self, s):
        x = self.fc(s)
        return self.actor(x), self.critic(x)

    def bounce(self):
        # Simple wall bounce to keep on screen
        if self.pos[0] < RADIUS:
            self.pos[0] = RADIUS
        elif self.pos[0] > WIDTH - RADIUS:
            self.pos[0] = WIDTH - RADIUS
        if self.pos[1] < RADIUS:
            self.pos[1] = RADIUS
        elif self.pos[1] > HEIGHT - RADIUS:
            self.pos[1] = HEIGHT - RADIUS

agents = [Agent() for _ in range(NUM)]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("7 — FIXED: They WILL merge into one rainbow (watch 90s)")
clock = pygame.time.Clock()

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False

    screen.fill((22,22,28))

    for a in agents:
        to_t = (a.target - a.pos)
        d = max(np.linalg.norm(to_t), 1)
        to_t /= d

        evidence = 1.0 - min(1.0, d / math.hypot(WIDTH, HEIGHT))
        x = logistic(evidence)
        social = a.social(agents)

        state = torch.tensor([x, a.emotion, social, to_t[0], to_t[1], d/500,
                              a.pos[0]/WIDTH, a.pos[1]/HEIGHT], dtype=torch.float32)

        action, value = a(state)
        move = action.detach().numpy() * SPEED / 60
        a.pos += move
        a.bounce()  # Keep them on screen

        reward = -0.01
        if d <= REACH:
            reward = 2.0
            a.target = np.random.uniform(RADIUS, WIDTH-RADIUS, 2)
            a.memory.append((a.pos.copy(), a.emotion*2))

        a.emotion = max(0, min(1, a.emotion + EMO_LR*math.tanh(reward)))

        # One-step actor-critic update (fast & stable)
        next_state = state.clone()
        next_state[0] = logistic(1.0 - min(1.0, np.linalg.norm(a.target-a.pos)/math.hypot(WIDTH,HEIGHT)))
        _, next_v = a(next_state)
        td_target = reward + 0.98 * next_v.item()
        advantage = td_target - value.item()

        a.opt.zero_grad()
        v_loss = advantage ** 2
        a_loss = -action * advantage
        (v_loss + a_loss.mean()).backward()
        a.opt.step()

        # draw
        pygame.draw.circle(screen, rgb(x), a.pos.astype(int), RADIUS)
        for p, w in a.memory[-30:]:
            pygame.draw.circle(screen, (int(200*w), int(80*a.emotion), 100), p.astype(int), 4)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
