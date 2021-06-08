from gym.envs.registration import register 

register(id='Snake-v0', entry_point='snakebot.envs:SnakeEnv', max_episode_steps=5000) 
register(id='SnakeVelocity-v0', entry_point='snakebot.envs:SnakeEnvVelocity', max_episode_steps=5000) 
register(id='SnakeTorque-v0', entry_point='snakebot.envs:SnakeEnvTorque', max_episode_steps=5000) 
