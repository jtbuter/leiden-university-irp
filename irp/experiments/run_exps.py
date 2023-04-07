from subprocess import call
from sklearn.model_selection import ParameterGrid

grid = ParameterGrid({
    'gamma': [0.6],
    'exploration-rate': [0.8, 0.9, 1.0]
})

for param in grid:
    call([
        "python", "experiments/sahba_2008.py",
        f"--gamma={param['gamma']}",
        f"--exploration-rate={param['exploration-rate']}"
    ])