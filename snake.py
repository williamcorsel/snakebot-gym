import argparse
from ppo import PPO_model
import gym
from stable_baselines3 import PPO

import numpy as np
import torch
import time

envs = ["snakebot:Snake-v0", "snakebot:SnakeVelocity-v0", "snakebot:SnakeTorque-v0"]

def test(env_name, model, n_runs, mode, snake_length):
	env_eval = gym.make(env_name, render=True, snake_length=snake_length)
	env_eval.seed(42)
	
	for run in range(n_runs):
		step = 0
		
		done = False
		obs = env_eval.reset()
		while True:
			
			if mode == "manual":
				action = env_eval.env._snake.generate_sin_move()
			else:
			
				if model is not None:
					action, _ = model.predict(obs)
				else:
					action = [-0.5 for _ in range(env_eval.action_space.shape[0])]
			
			env_eval.render()
			
			obs, _, done, _ = env_eval.step(action)
			step += 1
			
			time.sleep(1. / 240.) # Sleep for real-time playback
			
			if done:
				break


def main(args):
	ppo = PPO_model(args.env_name)
	net_arch = None

	if args.use_feature_extractor:
		net_arch = [128, 128, dict(pi=[64, 64], vf=[64, 64])]

	print(f"Using netarch (None = Default): {net_arch}")

	if args.mode == "train":
		print(f"Train PPO on env {args.env_name}")
		ppo.train(args.snake_length, args.n_timesteps, args.n_envs, args.seed, args.log_dir, args.eval_freq, args.save_freq, args.eval_episodes, args.render_eval, net_arch=net_arch)
	else:
		model = None
		if args.model is not None:
			model = PPO.load(args.model)
		test(args.env_name, model, args.n_test_runs, args.mode, args.snake_length)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Make a snake robot move using RL")
	parser.add_argument("mode", type=str, choices=["train", "test", "manual"],
						help="Running mode")
	
	# Env args
	parser.add_argument("--env_name", type=str, default="snakebot:Snake-v0", choices=envs,
						help="Environment to run")
	parser.add_argument("--snake_length", type=int, default=5,
						help="Length of snake")
	
	# Train args
	parser.add_argument("--log_dir", type=str, default="checkpoints",
						help="Directory to store logging and checkpoint files")
	parser.add_argument("--n_timesteps", type=int, default=20000000,
						help="Total number of training timesteps")
	parser.add_argument("--n_envs", type=int, default=16,
						help="Number of training envs")
	parser.add_argument("--eval_freq", type=int, default=250000,
						help="Evaluation frequency")
	parser.add_argument("--save_freq", type=int, default=1000000,
						help="Model save frequency")
	parser.add_argument("--eval_episodes", type=int, default=10,
						help="Number of eval episodes per evaluation")
	parser.add_argument("--render_eval", action="store_true", 
						help="Whether or not to display evaluation games")

	parser.add_argument("--use_feature_extractor", action="store_true",
						help="whether to use a shared [128,128] feature extractor before the [64, 64] and [64,64] actor & critic networks"
	)
	
	# Test args
	parser.add_argument('--model', type=str, default="./models/ppo_5_20000000_steps", #Default to trained snake of length 5
						help="Trained model directory")
	parser.add_argument('--n_test_runs', type=int, default=10,
						help="Number of test runs to display")
	
	# Misc
	parser.add_argument("--seed", type=int, default=42,
						help="Random seed")
	
	args = parser.parse_args()
	
	np.random.seed(args.seed)
	torch.random.manual_seed(args.seed)

	main(args)