from irp.wrappers import ExpandDims, Discretize
from irp.envs import UltraSoundEnv, PaperUltraSoundEnv, Paper2008UltraSoundEnv
from irp.q import Q
from irp import utils

from gym.wrappers import TimeLimit
import numpy as np
import matplotlib.pyplot as plt

data = utils.make_sample_label("case10_11.png", "case10_10.png")
train_image, train_label = data[0]
test_image, test_label = data[1]

height, width = train_label.shape
lows = {'area': 0., 'compactness': 0., 'objects': 0.}
# highs = {'area': 1., 'compactness': 1., 'objects': 34}
highs = {'area': 1., 'compactness': 1., 'objects': np.ceil(width / 2) * np.ceil(height / 2)}
bins = (35, 35, 35)

env = PaperUltraSoundEnv(train_image, train_label, num_thresholds=15)
env = Discretize(env, lows, highs, bins)
env = TimeLimit(env, 50)

exploration_fraction = 0.2
exploration_rate = 0.05
learning_rate = 0.2
gamma = 0.95

model = Q(
    env,
    learning_rate=learning_rate,
    gamma=gamma,
    exploration_fraction=exploration_fraction,
    exploration_final_eps=exploration_rate,
    tensorboard_log="logs"
)

model.learn(200000, log_interval=1, tb_log_name="run")

env = PaperUltraSoundEnv(test_image, test_label, num_thresholds=15)
env = Discretize(env, lows, highs, bins)
env = TimeLimit(env, 10)

current_state = env.reset()
env.render()

done = False

while not False:
    action = model.predict(current_state, deterministic=True)

    next_state, reward, done, info = env.step(action)
    current_state = next_state

    plt.title((action, str(env.threshold_ids)))
    env.render()


# for i in range(5):
#     model = Q(
#         env,
#         learning_rate=learning_rate,
#         gamma=gamma,
#         exploration_fraction=exploration_fraction,
#         exploration_final_eps=exploration_rate,
#         tensorboard_log="logs"
#     )

#     model.learn(200000, log_interval=1, tb_log_name=f"run-200k-0.99-new_reward")

#     # env = PaperUltraSoundEnv(test_image, test_label, num_thresholds=10)
#     # env = Discretize(env, lows, highs, bins)
#     # env = TimeLimit(env, 150)

#     # current_state = env.reset()
#     # env.render()

#     for i in range(150):
#         continue

#         action = model.predict(current_state, deterministic=True)

#         print(action)

#         next_state, reward, done, info = env.step(action)
#         current_state = next_state

#         env.render()

# print('Done')