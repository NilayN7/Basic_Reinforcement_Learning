import random

REWARD = -1
DISCOUNT = 0.9
MAX_ERROR = 1e-3

NUM_ACTIONS = 4
ACTIONS = [(1,0), (0,1), (-1,0), (0,-1)]
NUM_ROWS = 3
NUM_COLS = 4
U = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
policy = [[random.randint(0,3) for i in range(NUM_COLS)] for i in range(NUM_ROWS)]

def get_value(U, r, c, action):
    change1, change2 = ACTIONS[action]
    new_r, new_c = (r + change1), (c + change2)
    if new_r < 0 or new_r > NUM_ROWS + 1 or new_c < 0 or new_c > NUM_COLS + 1:
        return U[r][c]
    else:
        return U[new_r][new_c]
    
def calculate_value(U, r, c, action):
    value = REWARD
    value += DISCOUNT * get_value(U, r, c, action)
    return value

def policyEvaluation(policy, U):
    while True:
        nextU = [[0, 0, 0,1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
        error = 0
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                if (r <= 1 and c == 3) or (r == c == 1):
                    continue
                nextU[r][c] = get_value(U, r, c, policy[r][c])
                error = max(error, abs(nextU[r][c] -  U[r][c]))
        U = nextU
        if error < MAX_ERROR:
            break
    return U

def policyIteration(policy, U):
    while True:
        U = policyEvaluation(policy, U)
        unchanged = True
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                if (r <= 1 and c == 3) or (r == c == 1):
                    continue
                maxAction, maxU = None, -float("inf")
                for action in range(NUM_ACTIONS):
                    u = calculate_value(U, r, c, action)
                    if u > maxU:
                        maxAction, maxU = action, u
                if maxU > calculate_value(U, r, c, policy[r][c]):
                    policy[r][c] = maxAction
                    unchanged = False
        if not unchanged:
            break
    return policy


