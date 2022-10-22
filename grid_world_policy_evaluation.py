import numpy as np

ACTIONS = {
    0: [1, 0]
    1: [-1, 0]
    2: [0, -1]
    3: [0, 1]
}

class GridWorld:
    def __init__(self, size=4):
        self.state_values = np.zeros((size, size))

    def reset(self):
        self.state_values = np.zeros((size, size))
        return
    
    def step (self, state, action):
        size = len(self.state_values) - 1
        if (state == (0,0)) or (state == (size, size)):
            return state, 0
        s_1 = (state[0] + action[0], state[1] + action[1])
        reward = -1
        if s_1[0] < 0 or s_1[0] >= len(self.state_values):
            s_1 = state
        elif s_1[1] < 0 or s_1[0] >= len(self.state_values):
            s_1 = state

        return s_1, reward

    def bellman_expectation(self, state, probs, discount):
        value = 0 
        for c, action in ACTION.items():
            s_1, reward = self.step(state, action)
            value += probs[c] * (reward + discount * self.state_values[s_1])
        return value

def policy_evaluation(env, policy=None, steps=1, discount=1.0, in_place=False):
    if policy is None:
        policy = np.ones((*env.state_values.shape, len(ACTIONS))) * 0.25

    for k in range(steps):
        values = env.state_values if in_place else np.zeros_like(env.state_values)
        for i in range(len(env.state_values)):
            for j in range(len(env.state_values[i])):
                state = (i, j)
                value = env.bellman_expectation(state, policy[i, j], discount)
                values[i, j] = value * discount
        env_state_values = values
    return env.state_values

################################################################################################################################
import numpy as np

ACTIONS = {
        0: [1, 0]    #up
        1: [-1, 0]   #down
        2: [0, -1]   #left
        3: [0, 1]    #right
    }

class State:
    def __init__(self, size=4):
        self.grid = np.zeros((size, size))

    def reset(self, size=4):
        self.grid = np.zeros((size, size))

    def step(self, state, action):
        size = len(self.grid)
        if (state == (0,0) or state == (size, size)):
            return state, 0
        s_1 = (state[0] + action[0], state[1] + action[1])
        reward = -1
        if (s_1[0] < 0 or s_1 >= len(self.grid)):
            s_1 = state
        elif (s_1[1] < 0 or s_1 >= len(self.grid)):
            s_1 = state

        return s_1 , reward

    def bellman_expectation(self, state, policy, discount):
        value = 0
        for c, action in ACTIONS.item():
            s_1, reward = self.step(state, action)
            value += policy[c] * (reward + discount * (self.grid[s_1]))
        return value

def policy_evaluation(env, policy=None, step=1, discount=1.0, in_place=False):
    if policy is None:
        policy = np.ones((*env.grid.shape, len(ACTIONS))) * 0.25
    for step in range(step):
        values = env.grid if in_place else np.zeros_like(env.grid)
        for i in range(len(env.grid)):
            for j in range(len(env.grid[i])):
                state = (i, j)
                value = bellman_expectation(state, policy[i, j], discount)
                values[i, j] = value
        env.grid = values
    return env.grid