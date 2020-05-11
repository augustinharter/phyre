import sys
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
import random
import cv2
import numpy as np
from tqdm import trange
import pathlib
import sys
import json

FOLDER_NAME = "test"
NUM_ROLLOUTS = 100 if not len(sys.argv) == 2 else int(sys.argv[1])
TIME_PER_STEP = 100  #No real time value correspondance, only relative meaning

# INIT
pygame.init()
size = 256
screen = pygame.display.set_mode((size, size))
pygame.display.set_caption("Phyre Task 00002:015")
clock = pygame.time.Clock()

draw_options = pymunk.pygame_util.DrawOptions(screen)

# SETUP
# Space
def add_ball(space, radius, pos, color= (200, 0 , 0, 255)):
  mass = radius / 5
  moment = pymunk.moment_for_circle(mass, 0, radius)
  body = pymunk.Body(mass, moment)
  body.position = pos
  shape = pymunk.Circle(body, radius)
  shape.friction = 0.5
  shape.color = color
  space.add(body, shape)
  return shape

def setup_space():
  space = pymunk.Space()
  space.gravity = (0.0, -1000.0)
  # Walls
  wall_body = pymunk.Body(body_type = pymunk.Body.STATIC)
  wall_body.position = (0,0)
  wall_width = 1
  wd = pymunk.Segment(wall_body, (-1,      0), (size,      0), wall_width)
  wu = pymunk.Segment(wall_body, (-1, 1+size), (size, 1+size), wall_width)
  wl = pymunk.Segment(wall_body, (-1,      0), (-1,     size), wall_width)
  wr = pymunk.Segment(wall_body, (size,    0), (size,   size), wall_width)
  space.add(wall_body, wl, wr, wu, wd)

  # Segments
  segment_width = 5
  floor_body = pymunk.Body(body_type = pymunk.Body.STATIC)
  wall_body.position = (0, 0)
  floor = pymunk.Segment(floor_body, (0, 3), (size, 3), segment_width)
  floor.color = (0, 0, 200, 255)
  floor.friction = 1

  plank_body = pymunk.Body(body_type = pymunk.Body.STATIC)
  plank_body.position = (0, 156)
  plank = pymunk.Segment(plank_body, (0, 0), (3*size//5, 0), segment_width)
  plank.color = (0, 0, 0, 255)
  plank.friction = 1

  space.add(floor_body, floor, plank_body, plank)

  # Ball
  #ball = add_ball(space, 16, (size//3, 7.5*size//8), color = (0, 200, 0, 255))
  #add_ball(space, 16, (size//3 - 5, 5.5*size//8)) # One Solution
  return space

# Collision
def pre_col(arbiter, space, data):
  global tags, contact
  sh = space.shapes
  tags = {1:sh[6], 2:sh[4], 6:sh[-1], 4:sh[5]}
  objs = {sh[6]:1, sh[4]:2, sh[-1]:6, sh[5]:4, sh[1]:0, sh[2]:0, sh[3]:0, sh[0]:0}
  a, b = arbiter.shapes
  if (a==tags[1] and b == tags[6]) or (b==tags[1] and a == tags[6]):
    contact += 1
  return True

def sep_col(arbiter, space, data):
  global tags, contact
  sh = space.shapes
  tags = {1:sh[6], 2:sh[4], 6:sh[-1], 4:sh[5]}
  objs = {sh[6]:1, sh[4]:2, sh[-1]:6, sh[5]:4, sh[1]:0, sh[2]:0, sh[3]:0, sh[0]:0}
  a, b = arbiter.shapes
  if a==tags[1] or b==tags[1]:
    # Check goal condition
    if b == tags[2] or a == tags[2]:
      if contact <=100:
        contact = 0


# Simulation
def find_solving_action(space, pos, radius):
  global size
  while True:
    action_radius = 16 + (random.random()-0.5) * 16
    action_pos = ((random.random()-0.5)*2*(radius+action_radius-2)+pos[0], (3*size//5) + random.random()*2*size//5)
    if not space.point_query_nearest(action_pos, action_radius+5, []):
      add_ball(space, action_radius, action_pos)
      break
  return action_pos, action_radius

def simulate(space, path):
  global tags, contact, count
  sh = space.shapes
  tags = {1:sh[6], 2:sh[4], 6:sh[-1], 4:sh[5]}
  screens = []
  positions = []
  interaction_step = 0

  frames = 0
  for step in range(350):
    for event in pygame.event.get():
      if event.type == QUIT:
        sys.exit(0)
      elif event.type == KEYDOWN and event.key == K_ESCAPE:
        sys.exit(0)
      
    space.step(TIME_PER_STEP/7000)

    screen.fill((255,255,255))
    space.debug_draw(draw_options)
    pygame.display.flip()
    screens.append(np.moveaxis(np.array(pygame.surfarray.array3d(screen)), 0,1)) 
    positions.append((tuple(tags[1].body.position), tuple(tags[6].body.position)))

    if contact>0:
      if step<10:
        return False
      if contact == 1:
        interaction_step = step
        contact +=1
        for i in range(-10, 0):
          try:
            cv2.imwrite(path+f"/{frames}.jpg", cv2.cvtColor(screens[step+i], cv2.COLOR_RGB2BGR))
          except IndexError:
            return False
          frames +=1

      cv2.imwrite(path+f"/{frames}.jpg", cv2.cvtColor(screens[step], cv2.COLOR_RGB2BGR))
      frames +=1
      if frames>=20:
        fp = open(path+f"/positions.txt", mode="w")
        json.dump(positions[interaction_step-10:interaction_step+10], fp)
        return True
  return False
    #clock.tick(50)

space_init = setup_space()
handler = space_init.add_default_collision_handler()
handler.pre_solve = pre_col

# MAIN LOOP
count = 0
while True:
  contact = 0
  space = space_init.copy()
  pos = (size/3, size*0.8 + (random.random()-0.5)*80)
  radius = 16 + (random.random()-0.5) * 16
  ball = add_ball(space, radius, pos, color = (0, 200, 0, 255))
  action_pos, action_radius = find_solving_action(space, pos, radius)

  # SIMULATION
  print("simulating...", count)
  path = f"rollouts/{FOLDER_NAME}/{count}"
  pathlib.Path(path).mkdir(parents=True, exist_ok=True)
  if simulate(space, path):
    count += 1
  if count>=NUM_ROLLOUTS:
    break
