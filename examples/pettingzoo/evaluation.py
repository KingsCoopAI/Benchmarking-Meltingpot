import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gym
import wandb
import socket
from meltingpot import substrate
import stable_baselines3
from stable_baselines3.common import callbacks
from stable_baselines3.common import torch_layers
from stable_baselines3.common import vec_env
from stable_baselines3.independent_ppo import IndependentPPO
import supersuit as ss
import torch
from torch import nn
import torch.nn.functional as F
import argparse
import numpy as np
import random
import utils
from wrappers.transform import obs2attr
import time
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def parse_args():
    parser = argparse.ArgumentParser("Stable-Baselines3 PPO with Parameter Sharing")
    parser.add_argument(
        "--env-name",
        type=str,
        default="coins",
        choices=['factory_commons__either_or', 'territory__inside_out', 'clean_up', 'chemistry__three_metabolic_cycles', 'chicken_in_the_matrix__repeated', 'chemistry__two_metabolic_cycles_with_distractors', 'territory__open', 'predator_prey__orchard', 'commons_harvest__open', 
                 'running_with_scissors_in_the_matrix__one_shot', 'pure_coordination_in_the_matrix__arena', 'predator_prey__open', 'boat_race__eight_races', 'stag_hunt_in_the_matrix__arena', 'collaborative_cooking__crowded', 'predator_prey__alley_hunt', 'commons_harvest__closed', 
                 'predator_prey__random_forest', 'pure_coordination_in_the_matrix__repeated', 'chicken_in_the_matrix__arena', 'gift_refinements', 'coop_mining', 'fruit_market__concentric_rivers', 'prisoners_dilemma_in_the_matrix__arena', 'rationalizable_coordination_in_the_matrix__repeated', 
                 'prisoners_dilemma_in_the_matrix__repeated', 'externality_mushrooms__dense', 'rationalizable_coordination_in_the_matrix__arena', 'bach_or_stravinsky_in_the_matrix__arena', 'bach_or_stravinsky_in_the_matrix__repeated', 'collaborative_cooking__asymmetric', 
                 'collaborative_cooking__cramped', 'paintball__king_of_the_hill', 'collaborative_cooking__forced', 'chemistry__two_metabolic_cycles', 'chemistry__three_metabolic_cycles_with_plentiful_distractors', 'paintball__capture_the_flag', 'commons_harvest__partnership', 
                 'hidden_agenda', 'collaborative_cooking__figure_eight', 'running_with_scissors_in_the_matrix__arena', 'collaborative_cooking__circuit', 'coins', 'stag_hunt_in_the_matrix__repeated', 'daycare', 'territory__rooms', 'running_with_scissors_in_the_matrix__repeated', 
                 'collaborative_cooking__ring', 'allelopathic_harvest__open'],
        help="The SSD environment to use",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="The number of agents",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=4,
        help="The number of cpus",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="The number of envs",
    )
    parser.add_argument(
        "--kl-threshold",
        type=float,
        default=0.01,
        help="The number of envs",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        default=1000,
        help="length of training rollouts AND length at which env is reset",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5e8,
        help="Number of environment timesteps",
    )
    parser.add_argument(
        "--use-collective-reward",
        type=bool,
        default=False,
        help="Give each agent the collective reward across all agents",
    )
    parser.add_argument(
        "--inequity-averse-reward",
        type=bool,
        default=False,
        help="Use inequity averse rewards from 'Inequity aversion \
            improves cooperation in intertemporal social dilemmas'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5,
        help="Advantageous inequity aversion factor",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Disadvantageous inequity aversion factor",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user_name", type=str, default="1160677229")
    parser.add_argument("--model", type=str, default='baseline')
    parser.add_argument("--alg", type=str, default='PPO', choices=['PPO', 'A2C'])
    parser.add_argument("--eval_freq", type=int, default=10000000)
    args = parser.parse_args()
    return args

class CustomCNN(torch_layers.BaseFeaturesExtractor):
  """Class describing a custom feature extractor."""

  def __init__(
      self,
      observation_space: gym.spaces.Box,
      features_dim=128,
      num_frames=6,
      fcnet_hiddens=(1024, 128),
  ):
    """Construct a custom CNN feature extractor.

    Args:
      observation_space: the observation space as a gym.Space
      features_dim: Number of features extracted. This corresponds to the number
        of unit for the last layer.
      num_frames: The number of (consecutive) frames to feed into the network.
      fcnet_hiddens: Sizes of hidden layers.
    """
    super().__init__(observation_space, features_dim)
    # We assume CxHxW images (channels first)
    # Re-ordering will be done by pre-preprocessing or wrapper

    self.conv = nn.Sequential(
        nn.Conv2d(
            num_frames * 3, num_frames * 3, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),  # 18 * 21 * 21
        nn.Conv2d(
            num_frames * 3, num_frames * 6, kernel_size=5, stride=2, padding=0),
        nn.ReLU(),  # 36 * 9 * 9
        nn.Conv2d(
            num_frames * 6, num_frames * 6, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),  # 36 * 7 * 7
        nn.Flatten(),
    )
    flat_out = num_frames * 6 * 7 * 7
    self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
    self.fc2 = nn.Linear(
        in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

  def forward(self, observations) -> torch.Tensor:
    # Convert to tensor, rescale to [0, 1], and convert from
    #   B x H x W x C to B x C x H x W
    if observations.shape[1] != 12:
      observations = observations.permute(0, 3, 1, 2)
    features = self.conv(observations)
    features = F.relu(self.fc1(features))
    features = F.relu(self.fc2(features))
    return features


def main(args):
  # Config
  set_seed(args.seed)
  str_model = args.model
  env_name = args.env_name
  env_config = substrate.get_config(env_name)
  env = utils.parallel_env(env_config)
  rollout_len = 1000
  num_agents = env.max_num_agents
  eval_freq = args.eval_freq
  # Training
  num_envs = args.num_envs  # number of parallel multi-agent environments
  # number of frames to stack together; use >4 to avoid automatic
  # VecTransposeImage
  num_frames = 4
  # output layer of cnn extractor AND shared layer for policy and value
  # functions
  features_dim = 128
  fcnet_hiddens = [1024, 128]  # Two hidden layers for cnn extractor
  ent_coef = 0.001  # entropy coefficient in loss
  batch_size = (rollout_len * num_envs // 2
               )  # This is from the rllib baseline implementation
  lr = 0.0001
  n_epochs = 30
  gae_lambda = 1.0
  gamma = 0.99
  target_kl = 0.01
  grad_clip = 40
  verbose = 3
  model_path = None  # Replace this with a saved model
  alg = args.alg

  eval_env = utils.parallel_env(
        max_cycles=rollout_len,
        env_config=env_config,
    )
  eval_env = ss.observation_lambda_v0(eval_env, lambda x, _: x["RGB"],
                                      lambda s: s["RGB"])
  eval_env = ss.frame_stack_v1(eval_env, num_frames)
  eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
  eval_env = ss.concat_vec_envs_v1(
      eval_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
  eval_env = vec_env.VecMonitor(eval_env)
  eval_env = vec_env.VecTransposeImage(eval_env, True)
  
  

  policy_kwargs = dict(
      features_extractor_class=CustomCNN,
      features_extractor_kwargs=dict(
          features_dim=features_dim,
          num_frames=num_frames,
          fcnet_hiddens=fcnet_hiddens,
      ),
      net_arch=[features_dim],
  )

  tensorboard_log = "./results/sb3/harvest_open_ppo_paramsharing"


  run = wandb.init(config=args,
                         project="MeltingPot_pytorch",
                         entity=args.user_name, 
                         notes=socket.gethostname(),
                         name=str(env_name) +"_"+ str(str_model) + "_" + str(args.seed),
                         group="Evaluation_" + str(env_name) +"_"+ str(str_model),
                         dir="./",
                         reinit=True)
  if alg == "PPO":
    model = IndependentPPO(
      "CnnPolicy",
      env=eval_env,
      learning_rate=lr,
      n_steps=rollout_len,
      batch_size=batch_size,
      n_epochs=n_epochs,
      gamma=gamma,
      gae_lambda=gae_lambda,
      ent_coef=ent_coef,
      max_grad_norm=grad_clip,
      target_kl=target_kl,
      policy_kwargs=policy_kwargs,
      tensorboard_log=tensorboard_log,
      verbose=verbose,
      num_agents=num_agents
  )

  save_path = './policy_model/sb3/' + str(env_name) + "_" + str(str_model) + "_" + str(args.seed)
  eval_model = IndependentPPO.load(path=save_path,
                                     num_agents=num_agents,
                                     env=eval_env,
                                     policy="CnnPolicy",
                                     n_steps=rollout_len,)

  total_timestep = 0
  initial_time = time.time()
  while total_timestep <= eval_freq:
    obs = eval_env.reset()
    done = np.array([False] * num_agents)
    ep_rew = []
    ep_l = 0
    while not done.any():
      ep_l += 1
      total_timestep += 1 * num_envs
      actions = []
      for i in range(num_agents):
        action, _ = eval_model.policies[i].predict(obs[i])
        actions.append(action)
      obs, reward, done, info = eval_env.step(np.array(actions))
      ep_rew.append(reward)
      if done.any():
        current_time = time.time()
        fps = total_timestep / (current_time - initial_time)
        for polid in range(num_agents):
            wandb.log({f"{polid}/fps": fps}, step=total_timestep)
            wandb.log({f"{polid}/ep_rew_mean": np.mean(np.array(ep_rew)[:,polid])}, step=total_timestep)
            wandb.log({f"{polid}/ep_len_mean": ep_l}, step=total_timestep)
            wandb.log({f"{polid}/time_elapsed": int(time.time() - initial_time)}, step=total_timestep)
            wandb.log({f"{polid}/total_timesteps": total_timestep}, step=total_timestep)
        wandb.log({"SW_ep_rew_mean": np.mean(np.array(ep_rew))}, step=total_timestep)
        wandb.log({"SW_ep_rew_total": np.mean(np.sum(np.array(ep_rew),axis=1))}, step=total_timestep)
        initial_time = current_time
if __name__ == "__main__":
  main(args=parse_args())

