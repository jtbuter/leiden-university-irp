import numpy as np

from irp.experiments.tile_coding.tile_coding import *

class ValueFunction:
    OBJECTS_BOUND = [0, 32]
    ACTIONS = [-1, 0, 1]


    def __init__(self, stepSize, numOfTilings=8, maxSize=2048):
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings

        # divide step size equally to each tiling
        self.stepSize = stepSize / numOfTilings  # learning rate for each tile

        self.hashTable = IHT(maxSize)

        # weight for each tile
        self.weights = np.zeros(maxSize)

        # position and velocity needs scaling to satisfy the tile software
        self.areaScale = self.numOfTilings
        self.compactnessScale = self.numOfTilings
        self.objectScale = self.numOfTilings / (self.OBJECTS_BOUND[1] - self.OBJECTS_BOUND[0])

    # get indices of active tiles for given state and action
    def getActiveTiles(self, area, compactness, objects, action):
        # I think positionScale * (area - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        activeTiles = tiles(self.hashTable, self.numOfTilings,
                            [self.areaScale * area, self.compactnessScale * compactness, self.objectScale * objects],
                            [action])
        return activeTiles

    # estimate the value of given state and action
    def value(self, area, compactness, objects, action):
        # if area == COMPACTNESS_BOUND[1]:
        #     return 0.0
        activeTiles = self.getActiveTiles(area, compactness, objects, action)

        return np.sum(self.weights[activeTiles])

    # learn with given state, action and target
    def update(self, area, compactness, objects, action, target):
        activeTiles = self.getActiveTiles(area, compactness, objects, action)
        estimation = np.sum(self.weights[activeTiles])
        delta = self.stepSize * (target - estimation)
        for activeTile in activeTiles:
            self.weights[activeTile] += delta

    # get the # of steps to reach the goal under current state value function
    def costToGo(self, position, velocity):
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        return -np.max(costs)
