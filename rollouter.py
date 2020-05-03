import sys
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
import random
from tqdm import trange

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
  floor.color = (0, 0, 180, 255)
  floor.friction = 0.5

  plank_body = pymunk.Body(body_type = pymunk.Body.STATIC)
  plank_body.position = (0, 156)
  plank = pymunk.Segment(plank_body, (0, 0), (3*size//5, 0), segment_width)
  plank.color = (0, 0, 0, 255)
  plank.friction = 0.5

  space.add(floor_body, floor, plank_body, plank)

  # Ball
  ball = add_ball(space, 16, (size//3, 7.5*size//8), color = (0, 200, 0, 255))
  #add_ball(space, 16, (size//3 - 5, 5.5*size//8)) # One Solution
  return space

# Collision
def begin_col(arbiter, space, data):
  global tags, solved
  sh = space.shapes
  tags = {1:sh[6], 2:sh[4], 6:sh[-1], 4:sh[5]}
  a, b = arbiter.shapes
  if (a == tags[1] and b == tags[2]) or (a == tags[2] and b == tags[1]):
    solved = True
    print("solved!")

  return True

# Simulation
def find_solving_action(space):
  global size
  while True:
    action_pos = (random.random()*size, random.random()*size)
    action_radius = 5 + random.random()*size/4
    if not space.point_query_nearest(action_pos, action_radius, []):
      add_ball(space, action_radius, action_pos)
      break
  return action_radius, action_pos

def simulate(space, visual = False, apply_vel=None, apply_step=None):
  global tags, solved
  sh = space.shapes
  tags = {1:sh[6], 2:sh[4], 6:sh[-1], 4:sh[5]}

  if not apply_step:
    apply_vel = (random.random()*200-100, random.random()*200-100)
    apply_step = random.randint(1,100)
  rollout = {'pos':[], 'vel':[], 'apply_vel':apply_vel, 'apply_step':apply_step, 'collisions':[], 'solve_steps':None}

  b = tags[1]
  for step in range(350):
    for event in pygame.event.get():
      if event.type == QUIT:
        sys.exit(0)
      elif event.type == KEYDOWN and event.key == K_ESCAPE:
        sys.exit(0)

    if step == apply_step:
      b.body.velocity = apply_vel

    if not visual:
      rollout["pos"].append(b.body.position)
      rollout["vel"].append(b.body.velocity)

    space.step(1/70.0)

    if visual:
      screen.fill((255,255,255))
      space.debug_draw(draw_options)
      pygame.display.flip()
      clock.tick(50)

  return apply_vel, apply_step

space_init = setup_space()
handler = space_init.add_default_collision_handler()
handler.begin = begin_col

# MAIN LOOP
count = 0
while True:
  solved = False
  space = space_init.copy()
  #action_radius, action_pos = find_solving_action(space)

  # SIMULATION
  print("simulating...", count)
  apply_vel, apply_step = simulate(space, visual=False)
  print(apply_step, apply_vel)
  count += 1

  if solved:
    space = space_init.copy()
    sh = space.shapes
    tags = {1:sh[6], 2:sh[4], 6:sh[-1], 4:sh[5]}
    #add_ball(space, action_radius, action_pos)
    simulate(space, visual=True, apply_vel=apply_vel, apply_step= apply_step)
