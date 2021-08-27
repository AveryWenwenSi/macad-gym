#!/bin/env python
import gym
from gym.spaces import Box, Discrete
import macad_gym  # noqa F401

# import modules from macad_agents directly
import rllib
import a3c

from rllib.env_wrappers import wrap_deepmind
from rllib.models import register_mnih15_shared_weights_net

#native ray import 
import ray
from ray import tune
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune import run_experiments
from ray.tune.registry import register_env


# additional imports:
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

