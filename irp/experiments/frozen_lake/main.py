import gym
import os

from irp.callbacks import HParamCallback
from irp.wrappers import ExpandDims
from irp.manager import ExperimentManager
from irp.q import Q
import irp

def evaluate_policy(model: Q, env: gym.Env, episodes = 1000):
    misses = 0
    steps_list = []

    for episode in range(episodes):
        observation = env.reset()
        steps = 0

        while True:
            action = model.predict(observation)
            observation, reward, done, _ = env.step(action)
            steps += 1
            
            if done and reward == 1:
                # print('You have got the fucking Frisbee after {} steps'.format(steps))
                steps_list.append(steps)

                break
            elif done and reward == 0:
                # print("You fell in a hole!")
                misses += 1
                break

    print('Your success rate was {:.2f} %'.format((1 - (misses/episodes)) * 100))

def main(manager: ExperimentManager):
    experiments_folder = os.path.join(
        irp.ROOT_DIR, manager.experiment_root, manager.experiment_name
    )
    env = gym.make('FrozenLake-v1')
    env = ExpandDims(env)

    for lr in [0.05, 0.1, 0.15, 0.2]:
        for e in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for g in [0.90, 0.92, 0.94, 0.96, 0.98]:
                model = Q(
                    env, learning_rate=lr, gamma=g, exploration_fraction=e,
                    tensorboard_log=experiments_folder
                )

                model.learn(100000, tb_log_name=f'lr={lr},e={e},g={g}', callback=HParamCallback())

                evaluate_policy(model, env)

# Get the name of the current file
code_file = __file__

# Explicitly use path to current file, instead of relative
cfg_file = os.path.join(irp.ROOT_DIR, 'experiments/sahba2008/conf.yaml')

# Initialize the manager
manager = ExperimentManager(
    experiment_root='results', experiment_name='frozen_lake',
    code_file=code_file, tb_friendly=True, verbose=0
)

manager.start(main)



