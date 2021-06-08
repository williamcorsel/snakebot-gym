import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList


class PPO_model:
    def __init__(self, env_name) -> None:
        self.env_name = env_name
    

    def train(self, snake_length, n_timesteps, n_envs, seed, log_dir, eval_freq, save_freq, eval_episodes, render, net_arch= None):
        env = make_vec_env(self.env_name, n_envs=n_envs, env_kwargs={"render" : False, "snake_length" : snake_length})
        env.seed(seed)
        
        env_eval = make_vec_env(self.env_name, n_envs=1, env_kwargs={"render" : render, "snake_length" : snake_length})
        env_eval.seed(seed+1)
        
        model = PPO("MlpPolicy", env, verbose=1, n_epochs=10, n_steps=4096, seed=seed, policy_kwargs=dict(net_arch=net_arch))
        
        checkpoint_callback = CheckpointCallback(save_freq=save_freq // n_envs, save_path=log_dir, name_prefix=f"ppo_{snake_length}_arch_{'default' if net_arch is None else 'shared'}")
        eval_callback = EvalCallback(env_eval, best_model_save_path=log_dir, log_path=log_dir, eval_freq=eval_freq // n_envs, n_eval_episodes=eval_episodes, render=render, deterministic=True)
        callback = CallbackList([checkpoint_callback, eval_callback])
        
        model.learn(total_timesteps=n_timesteps, callback=callback)
        model.save(os.path.join(log_dir, f"final_model_{n_timesteps}"))
        
        