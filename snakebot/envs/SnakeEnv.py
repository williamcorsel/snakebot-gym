import logging
from math import hypot

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from gym.utils import seeding
from pybullet_utils import bullet_client as bc
from snakebot.src.SnakeRobot import SnakeRobot

logger = logging.getLogger(__name__)


"""
gym.Env: https://github.com/openai/gym/blob/a5a6ae6bc0a5cfc0ff1ce9be723d59593c165022/gym/core.py 
"""
class SnakeEnv(gym.Env):
	MODE = "position"
	WIN_DISTANCE = 1

	def __init__(self, snake_length=5, render=False, goal=(0,-10,0)):
		"""Initialize environment

		Args:
			snake_length (int, optional): Number of joints of the snake bot. Defaults to 5.
			render (bool, optional): Whether or not to render. Defaults to False.
			goal (tuple, optional): Position of the goal. Defaults to (0,-10,0).
		"""
		super(SnakeEnv, self).__init__()
		self._SNAKE_LENGTH = snake_length
		self._render = render
		self._goal = goal
		self._start_distance = hypot(*goal)
		
		self.action_space = spaces.Box(-1, +1, (snake_length,), dtype=np.float32)
		self.observation_space = spaces.Box(-np.inf, np.inf, (2 + snake_length * 2,), dtype=np.float32)
		
		self._snake = None
		self._client = None

		#=========================== Renderer ==========================
		self.rendered_img = None
		self.viewer = None 
		self._cam_dist = 5
		self._cam_pitch = -40
		self._cam_yaw = 180
		self._render_width = 1280
		self._render_height = 720
		

	def _create_goal(self):
		"""Place duck object to visualise the goal
		"""
		base_pos = list(self._goal)
		base_pos[1] -= 1

		visualShapeId = self._client.createVisualShape(
			shapeType=p.GEOM_MESH,
			fileName="duck.obj",
			flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
			rgbaColor=[1, 1, 1, 1],
			specularColor=[0.4, .4, 0],
			visualFramePosition=[0, -0.02, 0],
			physicsClientId=self._client._client
		)

		collisionShapeId = self._client.createCollisionShape(
			shapeType=p.GEOM_MESH,
			fileName="duck_vhacd.obj",
		)
	
		orientation = self._client.getQuaternionFromEuler((1.571, 0, -1.571))
		
		self._goalID = self._client.createMultiBody(baseMass=0,
			baseOrientation=orientation,
			baseCollisionShapeIndex=collisionShapeId,
			baseVisualShapeIndex=visualShapeId,
			basePosition=base_pos,
		)


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	

	def step(self, action):
		"""Execute one time step within the environment

		Args:
			action (list): List of torque amounts to assign to the links

		Returns:
			np.array: Current observation
			float: reward
			bool: Whether or not the environment is done
			dict: Extra information
		"""
	
		# Set motors based on action
		self._snake.set_motors(action)
  
		# Step the simulation
		self._client.stepSimulation()
		
		# Get new state from snake
		state = self._snake.get_state()
  
		# Calculate reward based on distance to goal
		reward = -hypot(self._goal[0] - state[0], self._goal[1] - state[1])

		# Check if snake is in range of goal
		in_range = hypot(state[0] - self._goal[0], state[1] - self._goal[1]) < self.WIN_DISTANCE
		
		done = False
		if in_range:
			done = True
			reward = 100
  
		return state, reward, done, {}
 

	def reset(self):
		"""Reset the state of the environment to an initial state
		"""
		if self._render:
			if self._client is not None:
				self._client.disconnect()

			self._client = bc.BulletClient(connection_mode=p.GUI)
			self._client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
		else:
			self._client = bc.BulletClient(connection_mode=p.DIRECT)
	
		self._client.resetSimulation()

		self._client.setAdditionalSearchPath(pybullet_data.getDataPath())

		# Floor
		plane = self._client.createCollisionShape(p.GEOM_PLANE)
		self._client.createMultiBody(0, plane)
		self._client.setRealTimeSimulation(0)
		self._client.setGravity(0, 0, -9.81)
		
		# Random position and orientation
		base_position = np.append(self.np_random.uniform(low=-0.5, high=0.5, size=2), 0.5)
		base_orientation = self._client.getQuaternionFromEuler((0, 0, self.np_random.uniform(low=-0.2, high=0.2)))
		
		# Build snake and goal
		self._snake = SnakeRobot(self._SNAKE_LENGTH, self._client, base_position, base_orientation, self.MODE)
		self._create_goal()

		# Step once before getting state
		self._client.stepSimulation()

		return self._snake.get_state()
		

	def render(self, _):
		WINDOW_WIDTH=1280

		if self.viewer is None:
			from gym.envs.classic_control import rendering as rendering
			self.viewer = rendering.SimpleImageViewer(maxwidth=WINDOW_WIDTH)
		
		base_pos=[0,0,0]
		if self._client and self._snake:
			base_pos = self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0]

		self._client.resetDebugVisualizerCamera(
			self._cam_dist,
			self._cam_yaw,
			self._cam_pitch,
			base_pos
		)
	
	def close(self):
		self._client.disconnect()


# Snake envs
class SnakeEnvVelocity(SnakeEnv):
	MODE = "velocity"
 
class SnakeEnvTorque(SnakeEnv):
	MODE = "torque"
