from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

from gym.spaces import Box, Discrete
import ray
from ray import tune

from ray.rllib.agents.dqn.dqn import DQNAgent
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.rllib.agents.ppo.ppo import PPOAgent
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph

from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune import run_experiments
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from env_wrappers import wrap_deepmind
# from models import register_mnih15_shared_weights_net
import gym
import macad_gym

parser = argparse.ArgumentParser()

parser.add_argument("--num-iters", type=int, default=20)
parser.add_argument(
    "--num-workers",
    default=2,
    type=int,
    help="Num workers (CPU cores) to use")
parser.add_argument(
    "--num-gpus", default=1, type=int, help="Number of gpus to use. Default=1")
parser.add_argument(
    "--sample-bs-per-worker",
    default=50,
    type=int,
    help="Number of samples in a batch per worker. Default=50")
parser.add_argument(
    "--train-bs",
    default=128,
    type=int,
    help="Train batch size. Use as per available GPU mem. Default=500")
parser.add_argument(
    "--envs-per-worker",
    default=1,
    type=int,
    help="Number of env instances per worker. Default=10")
parser.add_argument(
    "--notes",
    default=None,
    help="Custom experiment description to be added to comet logs")

# register_mnih15_shared_weights_net()
# model_name = "mnih15_shared_weights"

env_name = "HeteNcomIndePOIntrxMATLS1B2C1PTWN3-v0"
env = gym.make(env_name)
env_actor_configs = env.configs
num_framestack = env_actor_configs["env"]["framestack"]


def env_creator(env_config):
    import macad_gym
    env = gym.make("HeteNcomIndePOIntrxMATLS1B2C1PTWN3-v0")
    # Apply wrappers to: convert to Grayscale, resize to 84 x 84,
    # stack frames & some more op
    env = wrap_deepmind(env, dim=84, num_framestack=num_framestack)
    return env


register_env(env_name, lambda config: env_creator(config))


# Placeholder to enable use of a custom pre-processor
class ImagePreproc(Preprocessor):
    def _init_shape(self, obs_space, options):
        shape = (84, 84, 3)  # Adjust third dim if stacking frames
        return shape

    def transform(self, observation):
        return observation


ModelCatalog.register_custom_preprocessor("sq_im_84", ImagePreproc)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    obs_space = Box(0.0, 255.0, shape=(84, 84, 3))
    act_space = Discrete(9)

    model_config = {
        # Model and preprocessor options.
        "model": {
            "custom_preprocessor": "sq_im_84",
            "dim": 84,
            "free_log_std": False,  # if discrete_actions else True,
            "grayscale": True,
        },
        # env_config to be passed to env_creator
        "env_config": env_actor_configs
    }

    policy_graphs = {
        "ppo_policy": (PPOPolicyGraph, obs_space, act_space, model_config),
        "dqn_policy": (DQNPolicyGraph, obs_space, act_space, model_config),
    }

    def policy_mapping_fn(agent_id):
        if agent_id % 2 == 0:
            return "ppo_policy"
        else:
            return "dqn_policy"

    dqn_trainer = DQNAgent(
        env=env_name,
        config={
            "multiagent": {
                "policy_graphs": policy_graphs,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["dqn_policy"],
            },
            "gamma": 0.95,
            "n_step": 3,
            "sample_batch_size": args.sample_bs_per_worker,
            "train_batch_size": args.train_bs,
        })

    ppo_trainer = PPOAgent(
        env=env_name,
        config={
            "multiagent": {
                "policy_graphs": policy_graphs,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": [], #["ppo_policy"],
            },
            # disable filters, otherwise we would need to synchronize those
            # as well to the DQN agent
            "observation_filter": "NoFilter",
            "sample_batch_size": args.sample_bs_per_worker,
            "train_batch_size": args.train_bs,
        })

    # disable DQN exploration when used by the PPO trainer
    ppo_trainer.optimizer.foreach_evaluator(
        lambda ev: ev.for_policy(
            lambda pi: pi.set_epsilon(0.0), policy_id="dqn_policy"))

    # Start our actual experiment.
    victim_checkpoint = "~/Code/macad-gym/example-multiagent/PPO_HomoNcomIndePOIntrxMASS3CTWN3-v0/checkpoint_500"

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    results = tune.run(
        "PPO",
        stop=stop,
        config=config,
        verbose=1,
        restore=victim_checkpoint,
    )

    for i in range(args.num_iters):
        print("== Iteration", i, "==")

        # improve the DQN policy
        print("-- DQN --")
        print(pretty_print(dqn_trainer.train()))

        # improve the PPO policy
        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        # swap weights to synchronize
        dqn_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
        ppo_trainer.set_weights(dqn_trainer.get_weights(["dqn_policy"]))

    # def gen_policy():
    #     config = {
    #         # Model and preprocessor options.
    #         "model": {
    #             "custom_model": model_name,
    #             "custom_options": {
    #                 # Custom notes for the experiment
    #                 "notes": {
    #                     "notes": args.notes
    #                 },
    #             },
    #             # NOTE:Wrappers are applied by RLlib if custom_preproc is NOT
    #             # specified
    #             "custom_preprocessor": "sq_im_84",
    #             "dim": 84,
    #             "free_log_std": False,  # if args.discrete_actions else True,
    #             "grayscale": True,
    #             # conv_filters to be used with the custom CNN model.
    #             # "conv_filters": [[16, [4, 4], 2], [32, [3, 3], 2],
    #             # [16, [3, 3], 2]]
    #         },
    #         # preproc_pref is ignored if custom_preproc is specified
    #         # "preprocessor_pref": "deepmind",

    #         # env_config to be passed to env_creator
    #         "env_config": env_actor_configs
    #     }
    #     return (PPOPolicyGraph, obs_space, act_space, config)
    # run_experiments({
    #     "MA-PPO-SSUI3CCARLA": {
    #         "run": "PPO",
    #         "env": env_name,
    #         "stop": {
    #             "training_iteration": args.num_iters
    #         },
    #         "config": {
    #             "log_level": "DEBUG",
    #             "num_sgd_iter": 10,
    #             "multiagent": {
    #                 "policy_graphs": policy_graphs,
    #                 "policy_mapping_fn":
    #                 tune.function(lambda agent_id: agent_id),
    #             },
    #             "num_workers": args.num_workers,
    #             "num_envs_per_worker": args.envs_per_worker,
                # "sample_batch_size": args.sample_bs_per_worker,
                # "train_batch_size": args.train_bs
    #         },
    #         "checkpoint_freq": 500,
    #         "checkpoint_at_end": True,
    #     }
    # })
