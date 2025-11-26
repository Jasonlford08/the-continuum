"""
continuum_gui.py
Interactive laboratory for The 0–1 Continuum
Jason Lankford — November 2025
"""

import pygame
import numpy as np
import math
import random

# === CONFIG ===
WIDTH, HEIGHT = 1000, 700
FPS = 60

# Continuum parameters (tweak live!)
LOGISTIC_K = 6.0
LOGISTIC_E0 = 0.5
EMO_LR = 0.08
BASE_THRESHOLD = 0.8
GAMMA = 0.22
MEMORY_DECAY = 0.97

# Creature
class Creature:
    def __init__(self):
        self.pos = np.array([WIDTH//2, HEIGHT//2], dtype=float)
        self.emotion = 0.7
        self.target = None
        self.memory = []  # (pos, emotion, weight)

    def set_target(self, pos):
        self.target = np.array(pos, dtype=float)

    def logistic(self, evidence):
        return 1.0 / (1.0 + math.exp(-LOGISTIC_K * (evidence - LOGISTIC_E0)))

    def continuum_to_rgb(self, x):
        wl = 400 + 300 * x
        R = max(0, min(1, (wl-500)/200))
        G = max(0, min(1, 1-abs(wl-550)/150))
        B = max(0, min(1, (500-wl)/200))
        return (int(255*R), int(255*G), int(255*B))

    def update(self, dt, mouse_pos, mouse_pressed):
        if mouse_pressed:
            self.set_target(mouse_pos)

        evidence = 0.0
        if self.target is not None:
            d = np.linalg.norm(self.target - self.pos)
            evidence = 1.0 - min(1.0, d / math.hypot(WIDTH, HEIGHT))

        x = self.logistic(evidence)
        threshold = BASE_THRESHOLD - GAMMA * (self.emotion - 0.7)

        reward = -0.02
        moved = False

        if self.target and x >= threshold:
            dir = (self.target - self.pos)
            dist = np.linalg.norm(dir)
            if dist > 10:
                dir /= dist
                speed = 140
                self.pos += dir * min(speed * dt, dist)
                moved = True
            if dist <= 16:
                reward = 1.5
                self.target = None
                self.memory.append((self.pos.copy(), self.emotion))

        self.emotion = max(0, min(1, self.emotion + EMO_LR * math.tanh(reward)))

        # decay memory
        for m in self.memory:
            m = list(m)
            m[2] = m[2] * MEMORY_DECAY if len(m) > 2 else 1.0

        return x, reward

# Init
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("The 0–1 Continuum — Interactive Lab (Jason Lankford, 2025)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)
bigfont = pygame.font.SysFont("Arial", 28)

creature = Creature()
show_help = True

running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    mouse_pos = np.array(pygame.mouse.get_pos())
    mouse_pressed = pygame.mouse.get_pressed()[0]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                show_help = not show_help
            if event.key == pygame.K_r:
                creature = Creature()

    x, reward = creature.update(dt, mouse_pos, mouse_pressed)

    screen.fill((18, 18, 28))

    # Draw memory places
    for pos, emo in [(m[0], m[1]) for m in creature.memory[-100:]]:
        alpha = int(180 * emo)
        color = (alpha, int(70*emo), 120)
        pygame.draw.circle(screen, color, pos.astype(int), 8)
        pygame.draw.circle(screen, (alpha//2, 20, 80), pos.astype(int), 14, 2)

    # Draw target
    if creature.target is not None:
        pygame.draw.circle(screen, (200, 80, 80), creature.target.astype(int), 10)
        pygame.draw.circle(screen, (255, 150, 150), creature.target.astype(int), 16, 3)

    # Draw creature
    color = creature.continuum_to_rgb(x)
    pygame.draw.circle(screen, color, creature.pos.astype(int), 20)
    pygame.draw.circle(screen, (255, 255, 255), creature.pos.astype(int), 22, 3)

    # HUD
    emo_bar = int(300 * creature.emotion)
    pygame.draw.rect(screen, (60,60,60), (20, 20, 304, 24))
    pygame.draw.rect(screen, (30,180,255), (22, 22, emo_bar, 20))
    screen.blit(font.render(f"Emotion E = {creature.emotion:.3f}", True, (255,255,255)), (340, 20))

    screen.blit(font.render(f"Continuum x = {x:.3f}", True, color), (20, 60))
    screen.blit(font.render(f"Effective threshold = {BASE_THRESHOLD - GAMMA*(creature.emotion-0.7):.3f}", True, (200,200,200)), (20, 90))

    # Title
    screen.blit(bigfont.render("The 0–1 Continuum — Jason Lankford, age 17", True, (180,220,255)), (20, HEIGHT-80))

    if show_help:
        help_text = [
            "Click anywhere → set target",
            "Watch hesitation → courage → commitment",
            "Red places = past joy (it will return)",
            "H = toggle help   R = reset",
            "This is the source code of feeling."
        ]
        for i, line in enumerate(help_text):
            screen.blit(font.render(line, True, (150,150,180)), (WIDTH-360, 20 + i*30))

    pygame.display.flip()

pygame.quit()
