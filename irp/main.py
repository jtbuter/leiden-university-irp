from irp.wrappers import ExpandDims, Discretize
from irp.envs import UltraSoundEnv, PaperUltraSoundEnv
from irp.callback import CustomCallback
from irp.q import Q
from irp import utils

from gym.wrappers import TimeLimit
import numpy as np
import matplotlib.pyplot as plt

data = utils.make_simple_train_test("case10_11.png", "case10_10.png")
train_image, train_label = data[0]
test_image, test_label = data[1]

height, width = train_label.shape
lows = {'area': 0., 'compactness': 0., 'objects': 0.}
highs = {'area': 1., 'compactness': 1., 'objects': np.ceil(width / 2) * np.ceil(height / 2)}
bins = (35, 35, 35)

env = PaperUltraSoundEnv(train_image, train_label, num_thresholds=10)
env = Discretize(env, lows, highs, bins)
env = TimeLimit(env, 150)

exploration_fraction = 0.1
exploration_rate = 0.05
learning_rate = 0.1
gamma = 0.95
callback = CustomCallback()

model = Q(
    env,
    learning_rate=learning_rate,
    gamma=gamma,
    exploration_fraction=exploration_fraction,
    exploration_final_eps=exploration_rate,
    tensorboard_log="logs"
)

model.learn(150000, log_interval=1, tb_log_name=f"run-{exploration_fraction}-{exploration_fraction}", callback=callback)

env = PaperUltraSoundEnv(test_image, test_label, num_thresholds=10)
env = Discretize(env, lows, highs, bins)
env = TimeLimit(env, 150)

current_state = env.reset()
env.render()

for i in range(150):
    action = model.predict(current_state, deterministic=True)

    print(action)

    next_state, reward, done, info = env.step(action)
    current_state = next_state

    env.render()