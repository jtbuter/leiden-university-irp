import numpy as np

def build_qtable(environment):
    qtable = {}

    for ti in range(environment.n_thresholds):
        s = environment.reset(ti=ti)
        qtable[s] = [0., 0., 0.]

    return qtable


def learn(environment, parameters):
    # We re-initialize the Q-table
    qtable = build_qtable(environment)

    episodes = parameters['episodes']
    alpha = parameters['alpha']
    gamma = parameters['gamma']
    epsilon = parameters['epsilon']
    epsilon_decay = parameters['epsilon_decay']

    # List of outcomes to plot
    outcomes = []

    # Training
    for e in range(episodes):
        state = environment.reset()
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

            # Update Q(s,a)
            qtable[state][action] = qtable[state][action] + \
                                    alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state][action])
            
            # Update our current state
            state = new_state

            # If we have a reward, it means that our outcome is a success
            if reward:
                outcomes[-1] = "Success"

        # Update epsilon
        epsilon = max(epsilon - epsilon_decay, 0)

    return qtable