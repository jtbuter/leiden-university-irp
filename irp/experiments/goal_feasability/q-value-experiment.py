import numpy as np

reward = lambda s: (s == 1) * 1
step = lambda s, a: a

alpha = 0.1
gamma = 0.8

print(round(1 / (1 - gamma), 0))

Q = np.zeros((2, 2))

state = np.random.randint(0, 2)

for i in range(100000):
    if np.random.random() < 0.05: action = np.random.randint(0, 2)
    else: action = np.argmax(Q[state])

    next_state = step(state, action)
    r = reward(next_state)

    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (r + gamma * np.max(Q[next_state, :]))

    state = next_state

state = 0

print('Final q-table')
print(Q)

print(state)

for i in range(5):
    action = np.argmax(Q[state])
    next_state = step(state, action)

    print(next_state)

    state = next_state