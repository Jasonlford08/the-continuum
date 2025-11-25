"""
5_multi_agent_rule_based.py
Five creatures. One room. They teach each other where joy happened.
"""

import math, random, sys
import pygame
import numpy as np

# Parameters
WIDTH, HEIGHT = 900, 600
FPS = 60
NUM = 5
RADIUS = 12
SPEED = 110.0
REACH = 16.0
SOCIAL_RADIUS = 90.0

LOGISTIC_K = 6.0
LOGISTIC_E0 = 0.5
EMO_LR = 0.08
BASE_THRESHOLD = 0.8
GAMMA = 0.22
MEMORY_DECAY = 0.97

def logistic(e): return 1.0/(1.0+math.exp(-LOGISTIC_K*(e-LOGISTIC_E0)))
def rgb(x):
    wl = 400 + 300*x
    return (int(255*max(0,min(1,(wl-500)/200))),
            int(255*max(0,min(1,1-abs(wl-550)/150))),
            int(255*max(0,min(1,(500-wl)/200))))

def update_emotion(e, r): return max(0,min(1, e + EMO_LR*math.tanh(r)))

class Creature:
    def __init__(self, x, y):
        self.pos = np.array([x,y], dtype=float)
        self.emotion = 0.7
        self.target = None
        self.memory = []  # (pos, weight)

    def social_bias(self, others):
        bias = 0.0
        count = 0
        for o in others:
            if o is self: continue
            if np.linalg.norm(self.pos - o.pos) < SOCIAL_RADIUS:
                for m in o.memory:
                    bias += m[1]
                count += len(o.memory)
        return bias / max(1, count)

    def choose_target(self):
        if random.random() < 0.25 or not self.memory:
            self.target = np.random.uniform(50, WIDTH-50, 2)
        else:
            weights = np.array([w[1] for w in self.memory])
            weights /= weights.sum()
            chosen = self.memory[np.random.choice(len(self.memory), p=weights)][0]
            self.target = chosen + np.random.randn(2)*25

    def act(self, dt, others):
        if self.target is None: self.choose_target()

        evidence = 1.0 - min(1.0, np.linalg.norm(self.target-self.pos)/math.hypot(WIDTH,HEIGHT))
        x = logistic(evidence)
        social = self.social_bias(others)
        thr = BASE_THRESHOLD - GAMMA*(self.emotion-0.7) - 0.2*social

        reward = -0.01
        if x >= thr:
            dir = (self.target - self.pos)
            d = np.linalg.norm(dir)
            if d > 5:
                self.pos += dir/d * min(SPEED*dt, d)
            if d <= REACH:
                reward = 1.0
                self.target = None
                self.memory.append((self.pos.copy(), self.emotion))

        self.emotion = update_emotion(self.emotion, reward)
        for m in self.memory:
            m = (m[0], m[1]*MEMORY_DECAY)

        return x

# Main
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("5 — Multi-Agent Religion")
clock = pygame.time.Clock()

creatures = [Creature(random.uniform(100,800), random.uniform(100,500)) for _ in range(NUM)]

running = True
while running:
    dt = clock.tick(FPS)/1000.0
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        if e.type == pygame.MOUSEBUTTONDOWN:
            for c in creatures: c.target = np.array(e.pos, dtype=float)

    screen.fill((22,22,28))
    for c in creatures:
        x = c.act(dt, creatures)
        pygame.draw.circle(screen, rgb(x), c.pos.astype(int), RADIUS)
        for pos, w in c.memory[-40:]:
            a = int(180 * w)
            pygame.draw.circle(screen, (a, int(60*c.emotion), 80), pos.astype(int), 4)

    pygame.display.flip()

pygame.quit()
sys.exit()
