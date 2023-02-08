# Brief description:
# This is the multiple 4-joints walker robot environment modified from bipedal walker.
#
# Task:
# The task of this environment is to carry a package cooperatively.
# Meanwhile, agents can also evolve themselves.
#


import sys, math
import numpy as np
import Box2D
import copy
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import gym
from bipedalwalker import AugmentBipedalWalker
from gym import spaces
from gym.utils import colorize, seeding

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

PACKAGE_POLY = [(-120, 5), (120, 5), (120, -5), (-120, -5)]
PACKAGE_LENGTH = 240

BIPED_LIMIT = 1600
BIPED_HARDCORE_LIMIT = 2000

HULL_FD = fixtureDef(shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
                     density=5.0,
                     friction=0.1,
                     categoryBits=0x0020,
                     maskBits=0x001,  # collide only with ground
                     restitution=0.0)  # 0.99 bouncy

WALKER_SEPERATION = 10  # in steps

# new parameters
U = 1.0 / SCALE
LEG_DOWN = - 8 * U
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

MAX_AGENTS = 40


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        # if walkers fall on ground
        for i, walker in enumerate(self.env.walkers):
            if walker.hull == contact.fixtureA.body:
                if self.env.package != contact.fixtureB.body:
                    self.env.fallen_walkers[i] = True
            if walker.hull == contact.fixtureB.body:
                if self.env.package != contact.fixtureA.body:
                    self.env.fallen_walkers[i] = True

        # if package is on the ground
        if self.env.package == contact.fixtureA.body:
            if contact.fixtureB.body not in [w.hull for w in self.env.walkers]:
                self.env.game_over = True
        if self.env.package == contact.fixtureB.body:
            if contact.fixtureA.body not in [w.hull for w in self.env.walkers]:
                self.env.game_over = True

            #    self.env.game_over = True
        for walker in self.env.walkers:
            for leg in [walker.legs[1], walker.legs[3]]:
                if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                    leg.ground_contact = True

    def EndContact(self, contact):
        for walker in self.env.walkers:
            for leg in [walker.legs[1], walker.legs[3]]:
                if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                    leg.ground_contact = False



class Agent(object):
    def __new__(cls, *args, **kwargs):
        agent = super(Agent, cls).__new__(cls)
        return agent

    @property
    def observation_space(self):
        raise NotImplementedError()

    @property
    def action_space(self):
        raise NotImplementedError()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class SingleEvoBipedalWalker(Agent):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': FPS}

    hardcore = False
    smalllegs = False
    talllegs = False

    def __init__(self, world, init_x=TERRAIN_STEP * TERRAIN_STARTPAD / 2,
                 init_y=TERRAIN_HEIGHT + 2 * LEG_H, n_walkers=2, one_hot=False, augment_reward=True):
        self.world = world
        self._n_walkers = n_walkers
        self.one_hot = one_hot
        self.hull = None
        self.init_x = init_x
        self.init_y = init_y

        self._seed()

        self.scale_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
        self.augment_reward = augment_reward
        self.fd_polygon = fixtureDef(shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]), friction=FRICTION)
        self.fd_edge = fixtureDef(shape=edgeShape(vertices=[(0, 0), (1, 1)]), friction=FRICTION, categoryBits=0x0001)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _reset(self):
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

        # agent original position
        # init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2

        # agent position
        init_x = self.init_x
        init_y = TERRAIN_HEIGHT + np.maximum(leg1_h_top + leg1_h_bot, leg2_h_top + leg2_h_bot)

        self.hull = self.world.CreateDynamicBody(position=(init_x, init_y), fixtures=HULL_FD)
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

            leg = self.world.CreateDynamicBody(position=(init_x, init_y - leg_h_top / 2 - LEG_DOWN), angle=(i * 0.05),
                                               fixtures=fixtureDef(
                                                   shape=polygonShape(box=(leg_w_top / 2, leg_h_top / 2)), density=1.0,
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

        self.drawlist = self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0: return 1
                self.p2 = point
                self.fraction = fraction
                return 0

        self.lidar = [LidarCallback() for _ in range(10)]

    def augment_env(self, scale_vector):
        self.scale_vector = np.copy(np.array(scale_vector, dtype=float))

    def apply_action(self, action):
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

    def get_observation(self):
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

        return state

    @property
    def observation_space(self):
        # 24 original obs (joints, etc), 2 displacement obs for each neighboring walker, 3 for package, 1 ID
        idx = MAX_AGENTS if self.one_hot else 1  # TODO
        return spaces.Box(low=-np.inf, high=np.inf, shape=(24 + 4 + 3 + idx,))

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=1, shape=(4,))


class MultiEvoBipedalWalker(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': FPS}

    hardcore = False

    def __init__(self, n_walkers=1, position_noise=1e-3, angle_noise=1e-3, reward_mech='local',
                 forward_reward=1.0, fall_reward=-100.0, drop_reward=-100.0, terminate_on_fall=True,
                 one_hot=False):
        self.n_walkers = n_walkers
        self.position_noise = position_noise
        self.angle_noise = angle_noise
        self._reward_mech = reward_mech
        self.forward_reward = forward_reward
        self.fall_reward = fall_reward
        self.drop_reward = drop_reward
        self.terminate_on_fall = terminate_on_fall
        self.one_hot = one_hot
        self.timer = 0

        self.setup()

    def setup(self):
        self.seed()
        self.viewer = None
        self.render_mode = 'rgb_array'

        self.world = Box2D.b2World()
        self.terrain = None

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.start_x = [init_x + WALKER_SEPERATION * i * TERRAIN_STEP for i in range(self.n_walkers)]
        self.walkers = [SingleEvoBipedalWalker(self.world, init_x=sx, init_y=init_y, one_hot=self.one_hot) for sx in
                        self.start_x]

        self.package_scale = self.n_walkers / 1.75
        self.package_length = PACKAGE_LENGTH / SCALE * self.package_scale

        self.prev_shaping = np.zeros(self.n_walkers)
        self.prev_package_shaping = 0.0

        self.terrain_length = int(TERRAIN_LENGTH * self.n_walkers * 1 / 8.)

        self.fd_polygon = fixtureDef(shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]), friction=FRICTION)
        self.fd_edge = fixtureDef(shape=edgeShape(vertices=[(0, 0), (1, 1)]), friction=FRICTION, categoryBits=0x0001)

        self.reset()

    @property
    def agents(self):
        return self.walkers

    @property
    def reward_mech(self):
        return self._reward_mech

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.package)
        self.package = None

        for walker in self.walkers:
            walker._destroy()

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
                if i > TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [(x, y), (x + TERRAIN_STEP, y), (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                        (x, y - 4 * TERRAIN_STEP)]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [(p[0] + TERRAIN_STEP * counter, p[1]) for p in poly]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                poly = [(x, y), (x + counter * TERRAIN_STEP, y),
                        (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP), (x, y + counter * TERRAIN_STEP), ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [(x + (s * stair_width) * TERRAIN_STEP, y + (s * stair_height) * TERRAIN_STEP),
                            (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (s * stair_height) * TERRAIN_STEP),
                            (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_height) * TERRAIN_STEP),
                            (x + (s * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_height) * TERRAIN_STEP)]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
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
            poly = [(self.terrain_x[i], self.terrain_y[i]),
                    (self.terrain_x[i + 1], self.terrain_y[i + 1])]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
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
            poly = [(x + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                     y + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP))
                    for a in range(5)]
            x1 = min([p[0] for p in poly])
            x2 = max([p[0] for p in poly])
            self.cloud_poly.append((poly, x1, x2))

    def _generate_package(self):
        init_x = np.mean(self.start_x)
        init_y = TERRAIN_HEIGHT + 3 * LEG_H
        self.package = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x * self.package_scale / SCALE, y / SCALE) for x, y in PACKAGE_POLY]),
                density=1.0,
                friction=0.5,
                categoryBits=0x004,
                # maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.package.color1 = (205 / 256, 133 / 256, 63 / 256)
        self.package.color2 = (139 / 256, 69 / 256, 19 / 256)

    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.fallen_walkers = np.zeros(self.n_walkers, dtype=np.bool)
        self.prev_shaping = np.zeros(self.n_walkers)
        self.prev_package_shaping = 0.0
        self.scroll = 0.0
        self.lidar_render = 0

        self.timer = 0

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        self._generate_package()
        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        # scenario object
        self.drawlist = copy.copy(self.terrain)
        self.drawlist += [self.package]

        # bipedal walker objects
        for walker in self.walkers:
            walker._reset()
            self.drawlist += walker.legs
            self.drawlist += [walker.hull]

        return self.step(np.array([0, 0, 0, 0] * self.n_walkers))[0], {}

    def step(self, actions):
        act_vec = np.reshape(actions, (self.n_walkers, 4))
        assert len(act_vec) == self.n_walkers

        for i in range(self.n_walkers):
            self.walkers[i].apply_action(act_vec[i])

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        obs = [walker.get_observation() for walker in self.walkers]

        xpos = np.zeros(self.n_walkers)
        obs = []
        done = False
        rewards = np.zeros(self.n_walkers)

        for i in range(self.n_walkers):
            pos = self.walkers[i].hull.position
            x, y = pos.x, pos.y
            xpos[i] = x

            wobs = self.walkers[i].get_observation()
            nobs = []
            for j in [i - 1, i + 1]:
                # if no neighbor (for edge walkers)
                if j < 0 or j == self.n_walkers:
                    nobs.append(0.0)
                    nobs.append(0.0)
                else:
                    xm = (self.walkers[j].hull.position.x - x) / self.package_length
                    ym = (self.walkers[j].hull.position.y - y) / self.package_length
                    nobs.append(np.random.normal(xm, self.position_noise))
                    nobs.append(np.random.normal(ym, self.position_noise))
            xd = (self.package.position.x - x) / self.package_length
            yd = (self.package.position.y - y) / self.package_length
            nobs.append(np.random.normal(xd, self.position_noise))
            nobs.append(np.random.normal(yd, self.position_noise))
            nobs.append(np.random.normal(self.package.angle, self.angle_noise))
            # ID
            if self.one_hot:
                nobs.extend(np.eye(MAX_AGENTS)[i])
            else:
                nobs.append(float(i) / self.n_walkers)
            obs.append(np.array(wobs + nobs))

            # shaping = 130 * pos[0] / SCALE  # moving forward is a way to receive reward (normalized to get 300 on completion)
            # shaping -= 5.0 * abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished
            shaping = 0.0
            shaping -= 5.0 * abs(wobs[0])
            if self.prev_shaping[i] is not None:
                rewards[i] = shaping - self.prev_shaping[i]
            self.prev_shaping[i] = shaping
            for a in act_vec[i]:
                rewards[i] -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            if (self.walkers[i].smalllegs or self.walkers[i].talllegs) and self.walkers[i].augment_reward:  # augments reward according to design
                rewards[i] *= self.walkers[i].reward_factor

        package_shaping = self.forward_reward * 130 * self.package.position.x / SCALE
        rewards += (package_shaping - self.prev_package_shaping)
        self.prev_package_shaping = package_shaping

        self.scroll = xpos.mean() - VIEWPORT_W / SCALE / 5 - (self.n_walkers - 1) * WALKER_SEPERATION * TERRAIN_STEP

        terminated = False
        truncated = False
        if self.game_over or pos[0] < 0:
            rewards += self.drop_reward
            terminated = True
        if pos[0] > (self.terrain_length - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True
        rewards += self.fall_reward * self.fallen_walkers
        if self.terminate_on_fall and np.sum(self.fallen_walkers) > 0:
            terminated = True

        if self.hardcore:
            if self.timer >= BIPED_HARDCORE_LIMIT:
                truncated = True
        else:
            if self.timer >= BIPED_LIMIT:
                truncated = True

        self.timer += 1
        if self.reward_mech == 'local':
            return obs, rewards, terminated, truncated, {}
        return obs, [rewards.mean()] * self.n_walkers, terminated, truncated, {}

    def evo_env(self):
        for i in range(self.n_walkers):
            scale_vector = (1.0 + (np.random.rand(8) * 2 - 1.0) * 0.5)
            self.walkers[i].augment_env(scale_vector)

    def render(self, mode='rgb_array', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        render_scale = 0.75

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

        self.viewer.set_bounds(self.scroll,
                               VIEWPORT_W / SCALE * self.package_scale * render_scale + self.scroll,
                               0, VIEWPORT_H / SCALE * self.package_scale * render_scale)

        self.viewer.draw_polygon([(self.scroll, 0),
                                  (self.scroll + VIEWPORT_W / SCALE, 0),
                                  (self.scroll + VIEWPORT_W / SCALE, VIEWPORT_H / SCALE),
                                  (self.scroll, VIEWPORT_H / SCALE)], color=(0.9, 0.9, 1.0))

        self.viewer.draw_polygon([(self.scroll, 0),
                                  (self.scroll + VIEWPORT_W * self.package_scale / SCALE * render_scale, 0),
                                  (self.scroll + VIEWPORT_W * self.package_scale / SCALE * render_scale,
                                   VIEWPORT_H / SCALE * self.package_scale * render_scale),
                                  (self.scroll, VIEWPORT_H / SCALE * self.package_scale * render_scale)],
                                 color=(0.9, 0.9, 1.0))

        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2: continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE * self.package_scale: continue
            self.viewer.draw_polygon([(p[0] + self.scroll / 2, p[1]) for p in poly], color=(1, 1, 1))

        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE * self.package_scale: continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        for walker in self.walkers:
            if i < 2 * len(walker.lidar):
                l = walker.lidar[i] if i < len(walker.lidar) else walker.lidar[len(walker.lidar) - i - 1]
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


#
# if __name__ == "__main__":
#     # # Heurisic: suboptimal, have no notion of balance.
#     # env = MultiEvoBipedalWalker()
#     # # augment_vector = (1.0 + (np.random.rand(8) * 2 - 1.0) * 0.5)
#     # augment_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
#     # print("augment_vector", augment_vector)
#     # env.augment_env(augment_vector)
#     # env.reset()
#     # steps = 0
#     # total_reward = 0
#     # a = np.array([0.0, 0.0, 0.0, 0.0])
#     # STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
#     # SPEED = 0.29  # Will fall forward on higher speed
#     # state = STAY_ON_ONE_LEG
#     # moving_leg = 0
#     # supporting_leg = 1 - moving_leg
#     # SUPPORT_KNEE_ANGLE = +0.1
#     # supporting_knee_angle = SUPPORT_KNEE_ANGLE
#     # while True:
#     #     s, r, terminated, truncated, info = env.step(a)
#     #     total_reward += r
#     #     if steps % 20 == 0 or terminated:
#     #         print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
#     #         print("step {} total_reward {:+0.2f}".format(steps, total_reward))
#     #         print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4]]))
#     #         print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9]]))
#     #         print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
#     #     steps += 1
#     #
#     #     contact0 = s[8]
#     #     contact1 = s[13]
#     #     moving_s_base = 4 + 5 * moving_leg
#     #     supporting_s_base = 4 + 5 * supporting_leg
#     #
#     #     hip_targ = [None, None]  # -0.8 .. +1.1
#     #     knee_targ = [None, None]  # -0.6 .. +0.9
#     #     hip_todo = [0.0, 0.0]
#     #     knee_todo = [0.0, 0.0]
#     #
#     #     if state == STAY_ON_ONE_LEG:
#     #         hip_targ[moving_leg] = 1.1
#     #         knee_targ[moving_leg] = -0.6
#     #         supporting_knee_angle += 0.03
#     #         if s[2] > SPEED: supporting_knee_angle += 0.03
#     #         supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
#     #         knee_targ[supporting_leg] = supporting_knee_angle
#     #         if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
#     #             state = PUT_OTHER_DOWN
#     #     if state == PUT_OTHER_DOWN:
#     #         hip_targ[moving_leg] = +0.1
#     #         knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
#     #         knee_targ[supporting_leg] = supporting_knee_angle
#     #         if s[moving_s_base + 4]:
#     #             state = PUSH_OFF
#     #             supporting_knee_angle = min(s[moving_s_base + 2], SUPPORT_KNEE_ANGLE)
#     #     if state == PUSH_OFF:
#     #         knee_targ[moving_leg] = supporting_knee_angle
#     #         knee_targ[supporting_leg] = +1.0
#     #         if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * SPEED:
#     #             state = STAY_ON_ONE_LEG
#     #             moving_leg = 1 - moving_leg
#     #             supporting_leg = 1 - moving_leg
#     #
#     #     if hip_targ[0]: hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
#     #     if hip_targ[1]: hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
#     #     if knee_targ[0]: knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
#     #     if knee_targ[1]: knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]
#     #
#     #     hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head strait
#     #     hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
#     #     knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
#     #     knee_todo[1] -= 15.0 * s[3]
#     #
#     #     a[0] = hip_todo[0]
#     #     a[1] = knee_todo[0]
#     #     a[2] = hip_todo[1]
#     #     a[3] = knee_todo[1]
#     #     a = np.clip(0.5 * a, -1.0, 1.0)
#     #
#     #     env.render()
#     #     if terminated or steps == 3000: break

if __name__ == "__main__":
    n_walkers = 3
    reward_mech = 'local'
    env = MultiEvoBipedalWalker(n_walkers=n_walkers, reward_mech=reward_mech)
    env.evo_env()
    env.reset()
    for i in range(1000):
        env.render()
        a = np.array([env.agents[0].action_space.sample() for _ in range(n_walkers)])
        o, r, terminated, truncated, info = env.step(a)
        print("\nStep:", i)
        #print "Obs:", o
        print("Rewards:", r)
        #print "Term:", done
        if terminated:
            break