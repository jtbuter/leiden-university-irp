from irp.envs import Sahba2008UltraSoundEnv, TrusEnv
from irp.wrappers import Discretize, ActionMasker
from irp import utils, ROOT_DIR
from irp.mask_q import MaskedQ
from irp.q import Q
from irp.manager import ExperimentManager
from irp.callbacks import LogDissimilarityCallback, ActionDiversityCallback, EvalCallback

from stable_baselines3.common.callbacks import CallbackList

import os
import numpy as np

from gym.wrappers import TimeLimit

train_path = 'case10_11.png'
test_path = 'case10_12.png'

(train_x, train_y), (test_x, test_y) = utils.make_sample_label(train_path, test_path)

def action_mask(env: TrusEnv = None):
    return env.valid_action_mask()
    # return np.array([1, 1, 1, 1, 1], dtype=np.uint8)

def make_env(x, y, max_unimproved_timesteps, bins, max_timesteps):
    trus_env = Sahba2008UltraSoundEnv(x, y, num_thresholds=15, vjs=(0, 2, 5))
    discr_env = Discretize(trus_env, (0, 0, 0), (1, 1, bins[0] - 1), bins)
    env = TimeLimit(discr_env, max_timesteps)
    # env = ActionMasker(env, action_mask)

    return env

experiment_env = {
    'max_timesteps': (1e5, 'mt'),
    'bins': ((5, 3, 1), 'b'),
    'max_unimproved_timesteps': (1e5, 'mu')
}

env = make_env(train_x, train_y, **{key: value for key, (value, _) in experiment_env.items()})
eval_env = make_env(test_x, test_y, **{key: value for key, (value, _) in experiment_env.items()})

experiment_q = {
    'learning_rate': (0.8, 'lr'),
    'exploration_fraction': (0.5, 'ef'),
    'exploration_final_eps': (0.05, 'er'),
    'gamma': (0.6, 'g')
}

model = Q(
    env,
    tensorboard_log=os.path.join(ROOT_DIR, 'results/masked'),
    **{key: value for key, (value, _) in experiment_q.items()}
)

if isinstance(model, MaskedQ):
    assert isinstance(env, ActionMasker)
else:
    assert not isinstance(env, ActionMasker)

experiment = {**experiment_q, **experiment_env, **{'model': (model.__class__.__name__, 'mdl')}}
tb_log_name = ','.join([f"{token}={value}" for value, token in experiment.values()])

model.learn(25000, tb_log_name=tb_log_name, callback=CallbackList([
    LogDissimilarityCallback(), ActionDiversityCallback(), EvalCallback(eval_env, eval_freq=250, n_eval_timesteps=50)
]))
model.save(os.path.join(ROOT_DIR, 'results/masked/model_' + tb_log_name))

print(tb_log_name)