"""
2_memory_creature.py
The soul that remembers joy and pain
"""

import math, random, sys, time
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
MEMORY_DECAY = 0.98
MEMORY_MAX = 200

# Utilities
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

# Creature
class Creature:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.emotion = 0.7
        self.target = None
        self.memory = []  # {evidence, action, emotion, weight}

    def set_target(self, tx, ty):
        self.target = np.array([tx, ty], dtype=float)

    def sense_evidence(self):
        if self.target is None: return 0.0
        d = np.linalg.norm(self.target - self.pos)
        return 1.0 - min(1.0, d / math.hypot(WIDTH, HEIGHT))

    def update_memory(self, evidence, action, reward):
        weight = self.emotion * (0.5 + reward)
        self.memory.append({"evidence": evidence, "action": action, "emotion": self.emotion, "weight": weight})
        for m in self.memory:
            m["weight"] *= MEMORY_DECAY
        if len(self.memory) > MEMORY_MAX:
            self.memory = self.memory[-MEMORY_MAX:]

    def memory_bias(self):
        if not self.memory: return 0.0
        num = sum(m["evidence"] * m["weight"] for m in self.memory)
        den = sum(m["weight"] for m in self.memory)
        return num / den if den > 0 else 0.0

    def decide_and_act(self, dt):
        evidence = self.sense_evidence()
        x = logistic_mapping(evidence)
        thr = BASE_THRESHOLD - GAMMA * (self.emotion - 0.7) - 0.15 * self.memory_bias()
        action = False
        reward = -0.02

        if self.target and x >= thr:
            dir = self.target - self.pos
            dist = np.linalg.norm(dir)
            if dist > 1e-6:
                dir /= dist
                self.pos += dir * min(CREATURE_SPEED * dt, dist)
                action = True
            if dist <= REACH_DIST:
                reward = 1.0
                self.target = None

        self.emotion = emotional_update(self.emotion, reward)
        self.update_memory(evidence, int(action), reward)
        return x, thr

# Main loop
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2 — Memory Creature")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 14)

creature = Creature(WIDTH*0.2, HEIGHT*0.5)
creature.set_target(random.uniform(100,800), random.uniform(100,500))

running = True
while running:
    dt = clock.tick(FPS)/1000.0
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        if e.type == pygame.MOUSEBUTTONDOWN:
            creature.set_target(*e.pos)

    x, thr = creature.decide_and_act(dt)

    screen.fill((22,22,28))
    if creature.target:
        pygame.draw.circle(screen, (200,80,80), creature.target.astype(int), TARGET_RADIUS)
    pygame.draw.circle(screen, continuum_to_rgb(x), creature.pos.astype(int), CREATURE_RADIUS)

    # draw memory ghosts
    for mem in creature.memory[-50:]:
        alpha = int(200 * mem["weight"])
        color = (alpha, int(50*mem["emotion"]), 50)
        pos = creature.pos + np.random.randn(2)*8
        pygame.draw.circle(screen, color, pos.astype(int), 3)

    # HUD
    ew = int(200*creature.emotion)
    pygame.draw.rect(screen, (60,60,60), (10,10,204,18))
    pygame.draw.rect(screen, (30,160,200), (12,12,ew,14))
    screen.blit(font.render(f"E = {creature.emotion:.2f}", True, (220,220,220)), (220,10))

    pygame.display.flip()

pygame.quit()
sys.exit()
