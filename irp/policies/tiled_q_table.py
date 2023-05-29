import numpy as np
import matplotlib.pyplot as plt

class TiledQ():
    def __init__(self, n_features: int, n_actions: int, alpha: float):
        self.w = np.zeros((n_features, n_actions))

        self._n_features = n_features
        self._n_actions = n_actions
        self._alpha = alpha / n_features

        self.w: np.ndarray

    def predict(self, state: np.ndarray) -> int:
        values = self.values(state)

        return self._argmax(values)

    def values(self, state: np.ndarray) -> np.ndarray:
        values = [self.value(state, action) for action in range(self._n_actions)]

        return np.asarray(values)

    def value(self, state: np.ndarray, action: int) -> float:
        return self.w[state, action].sum()

    def update(self, state: np.ndarray, action: int, target: float):
        value = self.value(state, action)
        delta = self._alpha * (target - value)

        self.w[state, action] += delta

    def _argmax(self, values: np.ndarray) -> int:
        # Break ties, np.argmax() always defaults to picking index 0 for ties, this picks randomly
        return np.random.choice(np.flatnonzero(values == np.max(values)))

if __name__ == "__main__":
    reward = lambda s: int((s == np.array([0, 1, 2, 3])).all())
    step = lambda s, a: np.array([0, 1, 2, 3]) if a == 1 else np.array([0, 1, 3, 3])

    alpha = 0.1
    gamma = 0.95

    Q = TiledQ(n_features=4, n_actions=2, alpha=alpha)

    state = np.array([[0, 1, 3, 3], [0, 1, 2, 3]])[np.random.randint(0, 2)]

    q_values = []

    for i in range(200):
        for t in range(10):
            if np.random.random() < 0.3: action = np.random.randint(0, 2)
            else: action = Q.predict(state)

            next_state = step(state, action)

            r = reward(next_state)
            target = r + gamma * max(Q.values(next_state))
            
            Q.update(state, action, target)

            q_values.append(Q.value(state, action))

            state = next_state

        state = np.array([[0, 1, 3, 3], [0, 1, 2, 3]])[np.random.randint(0, 2)]

    N = 100

    plt.plot(np.convolve(q_values, np.ones(N) / N, mode='valid'))
    plt.show()
