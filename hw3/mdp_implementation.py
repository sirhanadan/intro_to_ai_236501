import copy
from copy import deepcopy
import numpy as np
from mdp import MDP


def get_next_state(mdp: MDP, state: tuple, action: str):
    row, col = state
    if action == "up":
        if row >= 1:
            if not mdp.board[row - 1][col] == "WALL":
                return (row - 1, col)
        else:
            return (row, col)
    if action == "down":
        if row < mdp.num_row - 1:
            if not mdp.board[row + 1][col] == "WALL":
                return (row + 1, col)
        else:
            return (row, col)
    if action == "right":
        if col < mdp.num_col - 1:
            if not mdp.board[row][col + 1] == "WALL":
                return (row, col + 1)
        else:
            return (row, col)
    if action == "left":
        if col >= 1:
            if not mdp.board[row][col - 1] == "WALL":
                return (row, col - 1)
        else:
            return (row, col)
    return (row, col)


def get_sum_of_probabilities(mdp: MDP, state: tuple, prev_utilities, action):
    row, col = state
    res = 0
    actions = ["up", "down", "right", "left"]
    next_states = [get_next_state(mdp, state, a) for a in actions]
    # note to self: next states holds the next states in the order they appear in actions
    probabilities = [(i, mdp.transition_function[action][i]) for i in range(4)]
    # ls = enumerate(mdp.transition_function[action])
    for i, prob_i in probabilities:
        next_state = next_states[i]
        if next_state == None:
            continue
        next_row = next_state[0]
        next_col = next_state[1]
        res += prob_i * prev_utilities[next_row][next_col]
    return res


def value_iteration(mdp: MDP, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    # new utility:
    new_utility = U_init
    # states:
    row_num = mdp.num_row
    col_num = mdp.num_col
    states = [(i, j) for i in range(row_num) for j in range(col_num)]
    # gamma:
    gamma = mdp.gamma
    # states with no updates to utility (walls and goal states):
    goal_states = mdp.terminal_states
    walls = []
    for state in states:
        if mdp.board[state[0]][state[1]] == 'WALL':
            walls.append(state)
    # actions:
    actions = mdp.actions
    # when to stop the function:
    termination_cond = (epsilon * (1 - gamma)) / gamma

    while True:
        d = 0  # delta
        prev_utility = copy.deepcopy(new_utility)
        for state in states:
            if state in walls:
                continue
            if state in goal_states:
                new_utility[state[0]][state[1]] = float(mdp.board[state[0]][state[1]])
                continue
            # Sum over s' for P(s'|a,s)*U(s'):
            all_prob_utils = [get_sum_of_probabilities(mdp, state, prev_utility, a) for a in actions]
            # max Sum ( P(s'|a,s)*U(s') ):
            max_prob_util = max(all_prob_utils)
            state_reward = float(mdp.board[state[0]][state[1]])  # rewards are saved as str in the board
            # U(s) = R(s) + gamma*max(Sum ( P(s'|a,s)*U(s') ))
            state_util = state_reward + gamma * max_prob_util
            new_utility[state[0]][state[1]] = state_util
            # update delta
            diff = abs(prev_utility[state[0]][state[1]] - new_utility[state[0]][state[1]])
            d = max(d, diff)
        if d < termination_cond:
            break
    # ========================
    return new_utility


def get_state_policy(mdp: MDP, state, utility):
    actions = mdp.actions
    best_action = "up"
    best_action_value = float('-inf')
    for action in actions:
        cur_action_value = get_sum_of_probabilities(mdp, state, utility, action)
        if best_action_value < cur_action_value:
            best_action_value = cur_action_value
            best_action = action
    return best_action


def get_policy(mdp: MDP, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Bellman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    # states:
    row_num = mdp.num_row
    col_num = mdp.num_col
    states = [(i, j) for i in range(row_num) for j in range(col_num)]
    policy = [[] for row in range(row_num)]
    for row in range(row_num):
        policy[row] = ['up' for i in range(col_num)]
    # states with no updates to utility (walls and goal states):
    goal_states = mdp.terminal_states
    walls = []
    for state in states:
        if mdp.board[state[0]][state[1]] == 'WALL':
            walls.append(state)
    for state in goal_states:
        policy[state[0]][state[1]] = mdp.board[state[0]][state[1]]
    for state in walls:
        policy[state[0]][state[1]] = mdp.board[state[0]][state[1]]
    for state in states:
        if state in goal_states:
            continue
        if state in walls:
            continue
        action = get_state_policy(mdp, state, U)
        policy[state[0]][state[1]] = action
    return policy
    # ========================


def get_state_cur_utility(mdp: MDP, utility, state, action):
    st_ut = 0.0
    actions = mdp.actions
    #action is the step we want to take, actual_action is the step actually taken which is not deterministic
    for i, actual_action in enumerate(actions):
        next_state = mdp.step(state, actual_action)
        prob = mdp.transition_function[action][i]
        next_state_ut = utility[next_state[0]][next_state[1]]
        st_ut += prob*next_state_ut
    return st_ut


def policy_evaluation(mdp: MDP, policy):
    # states:
    row_num = mdp.num_row
    col_num = mdp.num_col
    states = [(i, j) for i in range(row_num) for j in range(col_num)]
    utility = [[] for row in range(row_num)]
    for row in range(row_num):
        utility[row] = [0 for i in range(col_num)]
    # states with no updates to utility (walls and goal states):
    goal_states = mdp.terminal_states
    walls = []
    for state in states:
        if mdp.board[state[0]][state[1]] == 'WALL':
            walls.append(state)
    for state in goal_states:
        utility[state[0]][state[1]] = float(mdp.board[state[0]][state[1]])

    d, first = 0, True
    #sort of like ' do .. while()' instead of ' while() do .. ' #####THANK YOU MAROON FOR THIS IDEA
    while first or d > 0:
        d, first = 0, False
        for state in states:
            row = state[0]
            col = state[1]
            if state in walls:
                continue
            if state in goal_states:
                continue
            prev_ut = utility[row][col]
            new_sum_prob_ut = get_state_cur_utility(mdp, utility, state, policy[row][col])
            new_ut = float(mdp.board[row][col]) + mdp.gamma * new_sum_prob_ut
            utility[row][col] = new_ut
            d = max([d, abs(new_ut - prev_ut)])

    return utility


def get_state_policy_and_utility(mdp: MDP, state, utility):
    actions = mdp.actions
    best_action = "up"
    best_action_value = float('-inf')
    for action in actions:
        cur_action_value = get_sum_of_probabilities(mdp, state, utility, action)
        if best_action_value < cur_action_value:
            best_action_value = cur_action_value
            best_action = action
    return (best_action, best_action_value)

def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    # states:
    row_num = mdp.num_row
    col_num = mdp.num_col
    states = [(i, j) for i in range(row_num) for j in range(col_num)]
    goal_states = mdp.terminal_states
    walls = []
    for state in states:
        if mdp.board[state[0]][state[1]] == 'WALL':
            walls.append(state)

    #policy iteration alg:
    changed = True
    cur_policy = copy.deepcopy(policy_init)

    while changed:
        utility = policy_evaluation(mdp, cur_policy)
        changed = False
        for state in states:
            if state in walls:
                continue
            if state in goal_states:
                continue
            row = state[0]
            col = state[1]
            best_action , best_action_value = get_state_policy_and_utility(mdp, state, utility)

            st_ut = get_state_cur_utility(mdp, utility, state, cur_policy[row][col])

            if best_action_value > st_ut:
                cur_policy[row][col] = best_action
                changed = True
    # ========================
    return cur_policy
