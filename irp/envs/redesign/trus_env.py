import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

import gym
import gym.wrappers

# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.callbacks import CallbackList

from irp.envs.ultrasound.ultra_sound_env import UltraSoundEnv
# from irp.callbacks import (
#     LogNStepsCallback, EvalCallback, ActionDiversityCallback, LogDissimilarityCallback
# )
from irp.wrappers import Discretize
# from irp.q import Q
from irp import ROOT_DIR, utils

class TrusEnv(UltraSoundEnv):
    # Action map which allows for modifying the threshold index
    action_map = np.array([
        (-1, -1), (-1, 1),
        (1, -1), (1, 1),
        (0, 0) # Neutral action that doesn't modify
    ])

    def __init__(
        self,
        sample: np.ndarray,
        label: np.ndarray,
        num_thresholds: int,
        max_unimproved_steps: int = np.inf
    ):
        super().__init__(sample, label, num_thresholds)

        self.action_space = gym.spaces.Discrete(n=len(self.action_map))

        # Counts number of times we didn't improve the accuracy
        self.num_unimproved_steps = 0
        self.max_unimproved_steps = max_unimproved_steps

    def step(self, action: int):
        # Convert an action to new threshold indices
        new_threshold_ids = utils.process_thresholds(
            action, self.action_map, self.threshold_ids, self.num_thresholds
        )

        # # If the action we're trying to perform is not valid; do nothing
        # if not self._is_valid_action(*new_threshold_ids):
        #     new_threshold_ids = self.threshold_ids

        # Convert indices to gray-values
        lt, rt = self.thresholds[new_threshold_ids]

        # Extract a bit-mask using the gray-values
        bit_mask = utils.apply_threshold(self.sample, lt, rt)

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute dissimilarity and convert this to a reward
        dissim = utils.compute_dissimilarity(bit_mask, self.label)

        # We made an improvement
        if dissim < self.old_dissim:
            self.num_unimproved_steps = 0
        else:
            self.num_unimproved_steps += 1

        done = self._is_done(dissim)
        reward = self.reward(dissim)

        self.old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state

        info = {'dissimilarity': dissim}

        return np.asarray(next_state, dtype=np.float32), reward, done, info

    # def observation(self, bit_mask):
    #     return super().observation(bit_mask)[:2]

    def _is_done(self, dissim):
        if self.num_unimproved_steps >= self.max_unimproved_steps:
            return True

        return bool(dissim < 0.05)

    def _reward(self, dissim):
        if self.num_unimproved_steps >= self.max_unimproved_steps:
            return -1

        if dissim <= self.old_dissim:
            return 1
        elif dissim > self.old_dissim:
            return -1

    def reset(self):
        # Pick two random new threshold indices
        new_threshold_ids = np.random.choice(range(0, self.num_thresholds), 2)

        # Ensuring the left threshold is always smaller than the right threshold
        new_threshold_ids = np.sort(new_threshold_ids)

        # Convert indices to gray-values for generalization
        lt, rt = self.thresholds[new_threshold_ids]

        # Extract a bit-mask using the gray-values
        bit_mask = utils.apply_threshold(self.sample, lt, rt)

        # Convert the bit-mask to a discrete state
        next_state = self.observation(bit_mask)

        # Compute current dissimilarity
        dissim = utils.compute_dissimilarity(bit_mask, self.label)

        self.num_unimproved_steps = 0
        self.old_dissim = dissim
        self.threshold_ids = new_threshold_ids
        self.state = next_state

        return np.asarray(self.state, dtype=np.float32)

    def _render(self):
        lt, rt = self.thresholds[self.threshold_ids]

        # Extract a bit-mask using the gray-values
        state = utils.apply_threshold(self.sample, lt, rt)

        # Create a canvas to draw to
        fig, (label_axis, predict_axis) = plt.subplots(1, 2)

        # Show the images
        label_axis.imshow(self.label, cmap='gray', vmin=0, vmax=1)
        predict_axis.imshow(state, cmap='gray', vmin=0, vmax=1)

        # Set titles
        label_axis.title.set_text('Label')
        predict_axis.title.set_text('Prediction')

        plt.show()

if __name__ == "__main__":
    train_path = 'case10_11.png'
    test_path = 'case10_10.png'

    (train_x, train_y), (test_x, test_y) = utils.make_sample_label(train_path, test_path)
    max_unimproved_steps = 10
    max_timesteps = 100
    bins = (28, 28, 28)
    # max_objects = 256
    max_objects = bins[0] - 1
    lr = 0.1
    g = 0.6
    ef = 0.5
    er = 0.05
    tb_log_name = f'b={bins},mo={max_objects},er={er},lr={lr},g={g},ef={ef},mn={max_unimproved_steps},mt={max_timesteps}'

    trus_env = TrusEnv(
        train_x, train_y, num_thresholds=15, max_unimproved_steps=max_unimproved_steps
    )
    disc_env = Discretize(
        trus_env,
        lows={'area': 0.0, 'compactness': 0.0, 'objects': 0.0},
        highs={'area': 1.0, 'compactness': 1.0, 'objects': max_objects},
        bins=bins
    )
    env = gym.wrappers.TimeLimit(disc_env, max_timesteps)

    sample = trus_env.observation(bit_mask)
    print(sample, utils.discrete(sample, disc_env._state_bins))

    contours = utils.get_contours(bit_mask)
    biggest_object = max(contours, key=cv2.contourArea)

    bit_mask = np.ones((32, 16), dtype=np.uint8) * 255
    num_labels, area, largest_component = utils.get_largest_component(bit_mask)

    plt.imshow(largest_component, cmap='gray', vmin=0, vmax=1)
    plt.show()

    # print(sorted_components(components))


    # print(labels, stats.shape)

    # print(cv2.boundingRect(biggest_object))

    object_area = utils.get_area(biggest_object)
    d3_bitmask = np.dstack([bit_mask, bit_mask, bit_mask])

    cv2.drawContours(d3_bitmask, [biggest_object], 0, (0,255,0), 1)

    # plt.imshow(d3_bitmask)
    # plt.show()

    print(object_area)

    # eval_env = TrusEnv(
    #     test_x, test_y, num_thresholds=15, max_unimproved_steps=max_unimproved_steps
    # )
    # eval_env = Discretize(
    #     eval_env,
    #     lows={'area': 0.0, 'compactness': 0.0, 'objects': 0.0},
    #     highs={'area': 1.0, 'compactness': 1.0, 'objects': max_objects},
    #     bins=bins
    # )
    # eval_env = gym.wrappers.TimeLimit(eval_env, max_timesteps)

    # # Initialize potential callback
    # callback_list = []

    # # Log the reward every n steps
    # callback_list.append(LogNStepsCallback(freq=1000))

    # # Add the evaluation callback to the list of callbacks to execute
    # callback_list.append(EvalCallback(eval_env, eval_freq=5000, n_eval_episodes=100))

    # callback_list.append(ActionDiversityCallback())

    # callback_list.append(LogDissimilarityCallback())

    # # Set-up callback that executes all of the callbacks
    # callback = CallbackList(callbacks=callback_list)

    # model = Q(
    #     env,
    #     learning_rate=lr,
    #     gamma=g,
    #     exploration_fraction=ef,
    #     exploration_final_eps=er,
    #     tensorboard_log=os.path.join(ROOT_DIR, 'results/redesign'),
    # )

    # model.learn(
    #     100000,
    #     callback=callback,
    #     tb_log_name=tb_log_name,
    #     log_interval=None
    # )
    # model.save(
    #     os.path.join(
    #         ROOT_DIR, f'results/redesign/model_{tb_log_name}'
    #     )
    # )