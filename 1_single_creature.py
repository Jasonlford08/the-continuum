"""
1_single_creature.py
The first soul — born November 2025
Click to set target. Watch it hesitate, then commit.
"""

import math
import random
import sys
import time
import pygame
import numpy as np

# Parameters
WIDTH, HEIGHT = 900, 600
FPS = 60
CREATURE_RADIUS = 12
CREATURE_SPEED = 120.0
TARGET_RADIUS = 8
REACH_DIST = 14.0

# Continuum params
LOGISTIC_K = 6.0
LOGISTIC_E0 = 0.5

# Emotion params
EMO_LR = 0.08
BASE_THRESHOLD = 0.8
GAMMA = 0.22

# Utilities
def logistic_mapping(evidence, k=LOGISTIC_K, e0=LOGISTIC_E0):
    return 1.0 / (1.0 + math.exp(-k * (evidence - e0)))

def emotional_update(emotion, reward, lr=EMO_LR):
    new = emotion + lr * math.tanh(reward)
    return float(max(0.0, min(1.0, new)))

def continuum_to_rgb(x):
    wl = 400 + 300 * x
    R = max(0, min(1, (wl - 500) / 200))
    G = max(0, min(1, 1 - abs(wl - 550) / 150))
    B = max(0, min(1, (500 - wl) / 200))
    return (int(255 * R), int(255 * G), int(255 * B))

# Creature Class
class Creature:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.emotion = 0.7
        self.x_cont = 0.0
        self.target = None

    def set_target(self, tx, ty):
        self.target = np.array([tx, ty], dtype=float)

    def clear_target(self):
        self.target = None

    def sense_evidence(self):
        if self.target is None:
            return 0.0
        d = np.linalg.norm(self.target - self.pos)
        maxd = math.hypot(WIDTH, HEIGHT)
        evidence = 1.0 - min(1.0, d / maxd)
        return evidence

    def decide_and_act(self, dt):
        evidence = self.sense_evidence()
        self.x_cont = logistic_mapping(evidence)
        thr = BASE_THRESHOLD - GAMMA * (self.emotion - 0.7)
        action = False
        reward = 0.0

        if self.target is not None and self.x_cont >= thr:
            dir = self.target - self.pos
            dist = np.linalg.norm(dir)
            if dist > 1e-6:
                dir = dir / dist
                step = CREATURE_SPEED * dt
                move = dir * min(step, dist)
                self.pos += move
                action = True
            if np.linalg.norm(self.target - self.pos) <= REACH_DIST:
                reward += 1.0
                self.clear_target()
        else:
            reward -= 0.02

        if action and self.target is None:
            reward -= 0.3

        self.emotion = emotional_update(self.emotion, reward)
        return action, reward, evidence, thr

# Pygame main
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Continuum Creature Simulator — The First Soul")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 14)

    creature = Creature(WIDTH * 0.2, HEIGHT * 0.5)
    creature.set_target(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                creature.set_target(mx, my)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    creature.set_target(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
                if event.key == pygame.K_c:
                    creature.clear_target()

        action, reward, evidence, thr = creature.decide_and_act(dt)

        screen.fill((22, 22, 28))

        if creature.target is not None:
            pygame.draw.circle(screen, (200, 80, 80), creature.target.astype(int), TARGET_RADIUS)

        ccol = continuum_to_rgb(creature.x_cont)
        pygame.draw.circle(screen, ccol, creature.pos.astype(int), CREATURE_RADIUS)

        emo_w = int(200 * creature.emotion)
        pygame.draw.rect(screen, (60, 60, 60), (10, 10, 204, 18))
        pygame.draw.rect(screen, (30, 160, 200), (12, 12, emo_w, 14))
        screen.blit(font.render(f"Emotion E = {creature.emotion:.2f}", True, (220, 220, 220)), (220, 10))

        screen.blit(font.render(f"x (continuum) = {creature.x_cont:.3f}", True, (220, 220, 220)), (10, 36))
        screen.blit(font.render(f"thr (effective) = {thr:.3f}", True, (220, 220, 220)), (10, 56))
        screen.blit(font.render(f"evidence = {evidence:.3f}", True, (220, 220, 220)), (10, 76))
        screen.blit(font.render(f"action = {'MOVE' if action else 'WAIT'}  reward = {reward:.2f}", True, (220, 220, 220)), (10, 96))

        screen.blit(font.render("Click to set target. SPACE -> random. C -> clear.", True, (150, 150, 150)), (10, HEIGHT - 24))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
