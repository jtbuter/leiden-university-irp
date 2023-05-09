import numpy as np
import irp.wrappers as wrappers

def build_qtable(environment):
    dims = wrappers.utils.get_dims(environment.observation_space, environment.action_space)
    qtable = np.zeros(dims)

    return qtable

def learn(environment, parameters):
    # We re-initialize the Q-table
    qtable = build_qtable(environment)

    episodes = parameters['episodes']
    alpha = parameters['alpha']
    gamma = parameters['gamma']
    epsilon = parameters['epsilon']
    epsilon_decay = parameters['epsilon_decay']
    learning_delay = parameters['learning_delay']
    tilings = parameters['tilings']

    alpha /= tilings

    # List of outcomes to plot
    outcomes = []
    epsilons = []

    # Training
    for e in range(episodes):
        state = tuple(environment.reset())
        done = False

        # By default, we consider our outcome to be a failure
        outcomes.append("Failure")
        
        # Until the agent gets stuck in a hole or reaches the goal, keep training it
        while not done:
            # Generate a random number between 0 and 1
            rnd = np.random.random()

            # If random number < epsilon, take a random action
            if rnd < epsilon:
                action = environment.action_space.sample()
            # Else, take the action with the highest value in the current state
            else:
                action = np.argmax(qtable[state])

            # Implement this action and move the agent in the desired direction
            new_state, reward, done, info = environment.step(action)
            new_state = tuple(new_state)

            # Update Q(s,a)
            qtable[state][action] = qtable[state][action] + \
                                    alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state][action])
            
            # Update our current state
            state = new_state

            # If we have a reward, it means that our outcome is a success
            if reward:
                outcomes[-1] = "Success"

        if e >= learning_delay:
            # Update epsilon
            epsilon = max(epsilon - epsilon_decay, 0)

        epsilons.append(epsilon)

    return qtable, epsilons