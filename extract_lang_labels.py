import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# minimum working example to load a single OXE dataset
from pprint import pprint
from octo.data.oxe import make_oxe_dataset_kwargs
from octo.data.dataset import make_single_dataset

dataset_kwargs = make_oxe_dataset_kwargs(
    "bridge",
    "/home/marcelr/tensorflow_datasets",
)

dataset_kwargs["shuffle"] = False
dataset_kwargs["num_parallel_reads"] = 1
dataset_kwargs["num_parallel_calls"] = 1

pprint(dataset_kwargs)

dataset = make_single_dataset(
    dataset_kwargs,
    traj_transform_kwargs=dict(
        action_horizon=10,
    ),
    train=True,
)
iterator = dataset.iterator()

with open('language_instructions.txt', 'w') as f:
    for traj in tqdm(iterator, desc="Processing trajectories"):
        language_instructions = [a.decode("utf-8") for a in traj["task"]["language_instruction"]]
        for instruction in language_instructions:
            f.write(instruction + '\n')