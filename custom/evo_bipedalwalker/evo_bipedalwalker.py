import sys, math
import numpy as np

import Box2D
import torch
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

# import gymnasium as gym
# from gymnasium import spaces
# from gymnasium.utils import colorize, seeding

import gym
from gym import spaces
from gym.utils import colorize, seeding

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [
    (-30, +9), (+6, +9), (+34, +1),
    (+34, -8), (-30, -8)
]

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

BIPED_LIMIT = 1600
BIPED_HARDCORE_LIMIT = 2000

HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0)  # 0.99 bouncy


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.hull == contact.fixtureA.body or self.env.hull == contact.fixtureB.body:
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False


class EvoBipedalWalker(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': FPS}

    hardcore = False
    smalllegs = False
    talllegs = False

    def __init__(self, logger, augment_reward=True):
        self.cur_t = 0
        self.logger = logger
        self.stages = ['scale_transform', 'execution']
        self.stage = 'scale_transform'
        self.stage_ind = np.array([0])

        self.scale_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
        self.seed()
        self.viewer = None
        self.render_mode = 'rgb_array'

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None

        self.augment_reward = augment_reward

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
                                     friction=FRICTION)

        self.fd_edge = fixtureDef(shape=edgeShape(vertices=[(0, 0), (1, 1)]),
                                  friction=FRICTION,
                                  categoryBits=0x0001)

        high = np.array([np.inf] * 24)
        self.sim_action_space = spaces.Box(np.array([-1, -1, -1, -1]), np.array([+1, +1, +1, +1]))
        self.sim_obs_space = spaces.Box(-high, high)

        # dimension define
        self.scale_state_dim = self.scale_vector.size
        self.sim_obs_dim = self.sim_obs_space.shape[0]
        self.sim_action_dim = self.sim_action_space.shape[0]

        self.reset()
        self.timer = 0

    def if_use_transform_action(self):
        return ['scale_transform', 'execution'].index(self.stage)

    def reset(self):
        """ Reset the environment. """
        self.scale_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
        sim_obs, _ = self.execution_reset()
        self.stage = 'scale_transform'
        obs = [np.array([self.if_use_transform_action()]), self.scale_vector, sim_obs]
        self.cur_t = 0
        return obs, {}

    def step(self, action):
        self.cur_t += 1
        if self.stage == 'scale_transform':
            scale_state = action[:self.scale_state_dim]
            self.scale_vector = scale_state
            sim_obs, info = self.execution_reset()
            self.transit_execution()

            """ obs is a list, including three array: stage, scale_vector, sim_obs"""
            obs = [np.array([self.if_use_transform_action()]), self.scale_vector, sim_obs]

            reward = 0.0
            terminated = False
            truncated = False

            return obs, reward, terminated, truncated, {}

        elif self.stage == 'execution':
            scale_state = action[:self.scale_state_dim]
            assert (scale_state == self.scale_vector).all()
            action = action[self.scale_state_dim:]
            sim_obs, reward, terminated, truncated, info = self.execution_step(action)

            obs = [np.array([self.if_use_transform_action()]), self.scale_vector, sim_obs]
            return obs, reward, terminated, truncated, {}

        else:
            pass

    def execution_reset(self):
        self.augment_env(self.scale_vector)
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        self.timer = 0

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        # orig parameters
        # LEG_DOWN = -8/SCALE
        # LEG_W, LEG_H = 8/SCALE, 34/SCALE

        # new parameters
        U = 1.0 / SCALE
        LEG_DOWN = -8 * U

        # LEG_W = 1.0*8/SCALE
        # LEG_H = 1.0*34/SCALE # maybe make one for each leg?

        def calculate_total_area(x):
            return x[0] * x[1] + x[2] * x[3] + x[4] * x[5] + x[6] * x[7]

        def calculate_height(x):  # returns height of shorter leg
            return np.minimum(x[1] + x[3], x[5] + x[7])

        body_param = [8.0, 34.0, 6.4, 34.0, 8.0, 34.0, 6.4, 34.0]
        if self.smalllegs:
            self.orig_leg_area = calculate_total_area(body_param)
        if self.talllegs:
            self.orig_leg_height = calculate_height(body_param)

        for i in range(len(body_param)):
            body_param[i] = body_param[i] * self.scale_vector[i]

        if self.smalllegs:
            self.leg_area = calculate_total_area(body_param)
            self.reward_factor = 1.0 + np.log(self.orig_leg_area / self.leg_area)
        if self.talllegs:
            self.leg_height = calculate_height(body_param)
            self.reward_factor = 1.0 + np.log(self.leg_height / self.orig_leg_height)

        leg1_w_top = body_param[0] * U
        leg1_h_top = body_param[1] * U

        leg1_w_bot = body_param[2] * U
        leg1_h_bot = body_param[3] * U

        leg2_w_top = body_param[4] * U
        leg2_h_top = body_param[5] * U

        leg2_w_bot = body_param[6] * U
        leg2_h_bot = body_param[7] * U

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + np.maximum(leg1_h_top + leg1_h_bot, leg2_h_top + leg2_h_bot)
        self.hull = self.world.CreateDynamicBody(position=(init_x, init_y),
                                                 fixtures=HULL_FD)
        self.hull.color1 = (0.5, 0.4, 0.9)
        self.hull.color2 = (0.3, 0.3, 0.5)
        self.hull.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        self.legs = []
        self.joints = []

        for i in [-1, +1]:
            if i == -1:
                leg_w_top = leg1_w_top
                leg_w_bot = leg1_w_bot
                leg_h_top = leg1_h_top
                leg_h_bot = leg1_h_bot
            else:
                leg_w_top = leg2_w_top
                leg_w_bot = leg2_w_bot
                leg_h_top = leg2_h_top
                leg_h_bot = leg2_h_bot

            leg = self.world.CreateDynamicBody(position=(init_x, init_y - leg_h_top / 2 - LEG_DOWN),
                                               angle=(i * 0.05),
                                               fixtures=fixtureDef(
                                                   shape=polygonShape(box=(leg_w_top / 2, leg_h_top / 2)),
                                                   density=1.0,
                                                   restitution=0.0,
                                                   categoryBits=0x0020,
                                                   maskBits=0x001))
            leg.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            leg.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(bodyA=self.hull,
                                   bodyB=leg,
                                   localAnchorA=(0, LEG_DOWN),
                                   localAnchorB=(0, leg_h_top / 2),
                                   enableMotor=True,
                                   enableLimit=True,
                                   maxMotorTorque=MOTORS_TORQUE,
                                   motorSpeed=i,
                                   lowerAngle=-0.8,
                                   upperAngle=1.1)
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(position=(init_x, init_y - leg_h_top - leg_h_bot / 2 - LEG_DOWN),
                                                 angle=(i * 0.05),
                                                 fixtures=fixtureDef(
                                                     shape=polygonShape(box=(leg_w_bot / 2, leg_h_bot / 2)),
                                                     density=1.0,
                                                     restitution=0.0,
                                                     categoryBits=0x0020,
                                                     maskBits=0x001))
            lower.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            lower.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(bodyA=leg,
                                   bodyB=lower,
                                   localAnchorA=(0, -leg_h_top / 2),
                                   localAnchorB=(0, leg_h_bot / 2),
                                   enableMotor=True,
                                   enableLimit=True,
                                   maxMotorTorque=MOTORS_TORQUE,
                                   motorSpeed=1,
                                   lowerAngle=-1.6,
                                   upperAngle=-0.1)
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0

        self.lidar = [LidarCallback() for _ in range(10)]

        return self.execution_step(np.array([0, 0, 0, 0]))[0], {}

    def execution_step(self, action):
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
                 2.0 * self.hull.angularVelocity / FPS,
                 0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
                 0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
                 self.joints[0].angle,
                 # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
                 self.joints[0].speed / SPEED_HIP,
                 self.joints[1].angle + 1.0,
                 self.joints[1].speed / SPEED_KNEE,
                 1.0 if self.legs[1].ground_contact else 0.0,
                 self.joints[2].angle,
                 self.joints[2].speed / SPEED_HIP,
                 self.joints[3].angle + 1.0,
                 self.joints[3].speed / SPEED_KNEE,
                 1.0 if self.legs[3].ground_contact else 0.0]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 24

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        shaping = 130 * pos[
            0] / SCALE  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0 * abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        if (self.smalllegs or self.talllegs) and self.augment_reward:  # augments reward according to design
            reward *= self.reward_factor

        terminated = False
        truncated = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True

        if self.hardcore:
            if self.timer >= BIPED_HARDCORE_LIMIT:
                truncated = True
        else:
            if self.timer >= BIPED_LIMIT:
                truncated = True

        self.timer += 1
        return np.array(state), reward, terminated, truncated, {}

    def transit_execution(self):
        self.stage = 'execution'

    def augment_env(self, scale_vector):
        self.scale_vector = np.copy(np.array(scale_vector, dtype=float))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.integers(3, 5)
                poly = [
                    (x, y),
                    (x + TERRAIN_STEP, y),
                    (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                    (x, y - 4 * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [(p[0] + TERRAIN_STEP * counter, p[1]) for p in poly]
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.integers(1, 3)
                poly = [
                    (x, y),
                    (x + counter * TERRAIN_STEP, y),
                    (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                    (x, y + counter * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                stair_height = +1 if self.np_random.uniform(-1, 1) > 0.5 else -1
                stair_width = self.np_random.integers(4, 5)
                stair_steps = self.np_random.integers(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x + (s * stair_width) * TERRAIN_STEP, y + (s * stair_height) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (s * stair_height) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_height) * TERRAIN_STEP),
                        (x + (s * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_height) * TERRAIN_STEP),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(
                        fixtures=self.fd_polygon)
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    self.terrain.append(t)
                counter = stair_steps * stair_width

            elif state == STAIRS and not oneshot:
                s = stair_steps * stair_width - counter - stair_height
                n = s / stair_width
                y = original_y + (n * stair_height) * TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.integers(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1])
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge)
            color = (0.3, 1.0 if i % 2 == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly = []
        for i in range(TERRAIN_LENGTH // 20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
            y = VIEWPORT_H / SCALE * 3 / 4
            poly = [
                (x + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                 y + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP))
                for a in range(5)]
            x1 = min([p[0] for p in poly])
            x2 = max([p[0] for p in poly])
            self.cloud_poly.append((poly, x1, x2))

    def render(self, mode='rgb_array', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W / SCALE + self.scroll, 0, VIEWPORT_H / SCALE)

        self.viewer.draw_polygon([
            (self.scroll, 0),
            (self.scroll + VIEWPORT_W / SCALE, 0),
            (self.scroll + VIEWPORT_W / SCALE, VIEWPORT_H / SCALE),
            (self.scroll, VIEWPORT_H / SCALE),
        ], color=(0.9, 0.9, 1.0))
        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2: continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE: continue
            self.viewer.draw_polygon([(p[0] + self.scroll / 2, p[1]) for p in poly], color=(1, 1, 1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar) - i - 1]
            self.viewer.draw_polyline([l.p1, l.p2], color=(1, 0, 0), linewidth=1)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50 / SCALE
        x = TERRAIN_STEP * 3
        self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(0, 0, 0), linewidth=2)
        f = [(x, flagy2), (x, flagy2 - 10 / SCALE), (x + 25 / SCALE, flagy2 - 5 / SCALE)]
        self.viewer.draw_polygon(f, color=(0.9, 0.2, 0))
        self.viewer.draw_polyline(f + [f[0]], color=(0, 0, 0), linewidth=2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class EvoBipedalWalkerHardcore(EvoBipedalWalker):
    hardcore = True


class EvoBipedalWalkerSmallLegs(EvoBipedalWalker):
    smalllegs = True


class EvoBipedalWalkerHardcoreSmallLegs(EvoBipedalWalker):
    smalllegs = True


class EvoBipedalWalkerTallLegs(EvoBipedalWalker):
    talllegs = True
