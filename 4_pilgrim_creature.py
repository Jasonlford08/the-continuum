"""
4_pilgrim_creature.py
When no target exists, it quietly returns to the place of greatest past joy.
"""

import math, random, sys
import pygame
import numpy as np

# Parameters
WIDTH, HEIGHT = 900, 600
FPS = 60
CREATURE_RADIUS = 12
CREATURE_SPEED = 120.0
TARGET_RADIUS = 8
REACH_DIST = 14.0

LOGISTIC_K = 6.0
LOGISTIC_E0 = 0.5
EMO_LR = 0.08
BASE_THRESHOLD = 0.8
GAMMA = 0.22
MEMORY_DECAY = 0.97
MEMORY_MAX = 200

def logistic_mapping(evidence):
    return 1.0 / (1.0 + math.exp(-LOGISTIC_K * (evidence - LOGISTIC_E0)))

def emotional_update(emotion, reward):
    return float(max(0.0, min(1.0, emotion + EMO_LR * math.tanh(reward))))

def continuum_to_rgb(x):
    wl = 400 + 300 * x
    R = max(0, min(1, (wl-500)/200))
    G = max(0, min(1, 1-abs(wl-550)/150))
    B = max(0, min(1, (500-wl)/200))
    return (int(255*R), int(255*G), int(255*B))

class Creature:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.emotion = 0.7
        self.target = None
        self.memory = []  # {pos, weight, reward}

    def set_target(self, pos):
        self.target = np.array(pos, dtype=float)

    def return_home(self):
        if not self.memory:
            return
        weights = np.array([m["weight"] for m in self.memory])
        weights /= weights.sum()
        chosen = np.random.choice(self.memory, p=weights)
        noise = np.random.randn(2) * 30
        self.set_target(chosen["pos"] + noise)

    def sense_evidence(self):
        if self.target is None: return 0.0
        return 1.0 - min(1.0, np.linalg.norm(self.target - self.pos) / math.hypot(WIDTH, HEIGHT))

    def update_memory(self, reward):
        weight = self.emotion * (0.5 + max(0, reward))
        self.memory.append({"pos": self.pos.copy(), "weight": weight, "reward": reward})
        for m in self.memory:
            m["weight"] *= MEMORY_DECAY
        if len(self.memory) > MEMORY_MAX:
            self.memory = self.memory[-MEMORY_MAX:]

    def decide_and_act(self, dt):
        if self.target is None:
            self.return_home()

        evidence = self.sense_evidence()
        x = logistic_mapping(evidence)
        thr = BASE_THRESHOLD - GAMMA * (self.emotion - 0.7)

        reward = -0.015
        moved = False

        if self.target and x >= thr:
            dir = self.target - self.pos
            dist = np.linalg.norm(dir)
            if dist > 5:
                dir /= dist
                self.pos += dir * min(CREATURE_SPEED * dt, dist)
                moved = True
            if dist <= REACH_DIST:
                reward = 1.0
                self.target = None

        self.emotion = emotional_update(self.emotion, reward)
        self.update_memory(reward)
        return x

# Main
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("4 — Pilgrim Creature")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

creature = Creature(WIDTH//2, HEIGHT//2)

running = True
while running:
    dt = clock.tick(FPS)/1000.0
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        if e.type == pygame.MOUSEBUTTONDOWN:
            creature.set_target(e.pos)

    x = creature.decide_and_act(dt)

    screen.fill((22,22,28))
    if creature.target:
        pygame.draw.circle(screen, (200,80,80), creature.target.astype(int), TARGET_RADIUS)
    pygame.draw.circle(screen, continuum_to_rgb(x), creature.pos.astype(int), CREATURE_RADIUS)

    # sacred places
    for mem in creature.memory[-80:]:
        alpha = int(180 * mem["weight"])
        color = (alpha, int(60*mem["reward"]), 100) if mem["reward"] > 0 else (alpha//2, 20, 60)
        pygame.draw.circle(screen, color, mem["pos"].astype(int), 4)

    ew = int(200*creature.emotion)
    pygame.draw.rect(screen, (60,60,60), (10,10,204,18))
    pygame.draw.rect(screen, (30,160,200), (12,12,ew,14))
    pygame.display.flip()

pygame.quit()
sys.exit()
