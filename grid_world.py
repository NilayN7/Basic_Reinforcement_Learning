import numpy as np

NUM_ROWS = 3
NUM_COLS = 4
WIN_STATE = (0,3)
LOSE_STATE = (1,3)
DETERMINISTIC= True

class State:
    '''
    board:
      |  0   1   2   3
    --|-----------------
    0 |
    1 |
    2 |

    '''
    def __init__(self, state=START):
        self.board = np.zeros([NUM_ROWS, NUM_COLS])
        self.board[1, 1] = -1
        self.state = state
        self.is_end = False
        self.determine = DETERMINISTIC

    def give_reward(self):
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0
    
    def isEndFunc(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.is_end = True
    
    def next_position(self, action):
        if self.determine:
            if action == 'up':
                next_state = (self.state[0] - 1, self.state[1])
            elif action == 'down':
                next_state = (self.state[0] + 1, self.state[1])
            elif action == 'right':
                next_state = (self.state[0], self.state[1] + 1)
            else:
                self.state = (self.state[0], self.state[1] - 1)
            if (next_state[0] >= 0) and (next_state[0] <= NUM_ROWS - 1):
                if (next_state[1] >=0) and (next_state[1] <= NUM_COLS - 1):
                    if next_state != (1,1):
                        return next_state
            return self.state

class Agent:
    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3

        self.rewards = {}
        for i in range(NUM_COLS):
            for j in range(NUM_ROWS):
                self.state_values = [(i,j)] = 0

    def chooseAction(self):
        mx_next_reward = 0
        action = ""
        if np.random.uniform(0,1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            for a in self.actions:
                next_reward = self.state_values[self.State.next_position(a)]
                if next_reward >= mx_next_reward:
                    action = a
                    mx_next_reward = next_reward
        return action

    def takeAction(self, action):
        position = self.State.next_position(action)
        return State(state=position)
    
    def reset(self):
        self.states = []
        self.State = State()

    def Play(self, rounds=10):
        for i in range(rounds):
            if self.State.is_end:
                reward = self.State.give_reward()
                self.state_values[self.State.state] = reward
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                self.reset()
            else:
                action = self.chooseAction()
                self.states.append(self.State.next_position(action))
                self.State = self.takeAction(action)
                self.State.isEndFunc()