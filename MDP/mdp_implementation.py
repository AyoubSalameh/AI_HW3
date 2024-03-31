from copy import deepcopy
import numpy as np

#calculate utility for one action, but also tke in mind the probability in reaching others
def calculate_sum(mdp, U, i, j, action):
    dict = {'UP': 0, 'DOWN': 1, 'RIGHT': 2, 'LEFT': 3}
    sum = 0
    #the probabilitie in taking the current action
    prob = mdp.transition_function[action]
    for a in mdp.actions.keys():
        #getting the next step we reach if we take a
        i_next, j_next = mdp.step((i,j), a)
        sum += prob[dict[a]] * U[i_next][j_next]
    return sum

#TODO: not sure these two helpers work
def num_to_indices(mdp, state):
    row = state // mdp.num_col
    col = state % mdp.num_col
    return (row, col)

def indices_to_num(mdp, i, j):
    return i * mdp.num_col + j

def probability_src_to_dest(mdp, src, dest, policy):
    dict = {'UP': 0, 'DOWN': 1, 'RIGHT': 2, 'LEFT': 3}
    prob = 0
    src_idx = num_to_indices(mdp, src)
    policy_action = policy[src_idx[0]][src_idx[1]]
    if src_idx in mdp.terminal_states or mdp.board[src_idx[0]][src_idx[1]] == 'WALL':
        return 0
    dest_idx = num_to_indices(mdp, dest)
    for action in mdp.actions.keys():
        if dest_idx == mdp.step(src_idx, action):
            prob += mdp.transition_function[policy_action][dict[action]]

    return prob


#TODO: make sure if we should iterate over all states or over final states
def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U_tag = deepcopy(U_init)
    while True:
        delta = 0
        U = deepcopy(U_tag)
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                if mdp.board[i][j] == 'WALL':
                    continue
                elif (i, j) in mdp.terminal_states:
                    U_tag[i][j] = float(mdp.board[i][j])
                    continue
                else:
                    reward = mdp.board[i][j]
                    sums = [calculate_sum(mdp, U, i, j, action) for action in mdp.actions.keys()]
                    max_util = max(sums)
                    U_tag[i][j] = float(reward) + mdp.gamma * max_util
                if abs(U_tag[i][j] - U[i][j]) > delta:
                    delta = abs(U_tag[i][j] - U[i][j])

        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break
    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    pi = deepcopy(U)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == 'WALL' or (i,j) in mdp.terminal_states:
                pi[i][j] = None
            cur_max = float('-inf')
            for a in mdp.actions.keys():
                cur = calculate_sum(mdp, U, i, j, a)
                if cur > cur_max:
                    cur_max = cur
                    pi[i][j] = a
    return pi
    # ========================

#TODO: make sure np.array().T is a column vector
#TODO: know what to put in the matrices in wall/terminals
def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    U_ret = deepcopy(policy)
    n = mdp.num_row*mdp.num_col
    '''creating a reward vector'''

    rewards = []
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if(mdp.board[i][j] == 'WALL'):
                rewards.append(float('0'))
                continue
            rewards.append(float(mdp.board[i][j]))
    rewards = np.array(rewards).T

    '''creating probabilities matrix'''
    #nxn matrix
    trans = np.zeros((n, n))
    for dest in range(trans.shape[0]):
        for src in range(trans.shape[1]):
            #trans[src][dest] is the probability of going src -> dest
            trans[src][dest] = probability_src_to_dest(mdp, src, dest, policy)

    identity = np.identity(n)

    U = np.linalg.inv((identity - mdp.gamma * trans)) @ rewards
    for state, utility in enumerate(U):
        idx = num_to_indices(mdp, state)
        U_ret[idx[0]][idx[1]] = utility

    return U_ret
    # ========================

#TODO: make sure if we should iterate over all states or over final states
def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    while True:
        U = policy_evaluation(mdp, policy_init)
        unchanged = True
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                if mdp.board[i][j] == 'WALL' or (i,j) in mdp.terminal_states:
                    policy_init[i][j] = None
                    continue
                sums = [(calculate_sum(mdp, U, i, j, action), action) for action in mdp.actions.keys()]
                max_util = max(sums)
                curr_util = calculate_sum(mdp, U, i, j, policy_init[i][j])
                if curr_util < max_util[0]:
                    policy_init[i][j] = max_util[1]
                    unchanged = False
        if unchanged:
            return policy_init

    # ========================



"""For this functions, you can import what ever you want """


def get_all_policies(mdp, U):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    dict = {'UP': '↑', 'DOWN': '↓', 'RIGHT': '→', 'LEFT': '←'}
    pi = deepcopy(U)
    count = 0
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == 'WALL' or (i, j) in mdp.terminal_states:
                pi[i][j] = None
            cur_max = float('-inf')
            #print(i, j)
            for a in mdp.actions.keys():
                cur = calculate_sum(mdp, U, i, j, a)
                if cur >= cur_max:
                    if cur == cur_max:
                        #print("concatenating", dict[a], a)
                        count+=1
                        pi[i][j] = (pi[i][j]) + dict[a]
                        #print(pi[i][j])
                    else:
                        #print("replacing ׳ןאי", dict[a], a)
                        cur_max = cur
                        pi[i][j] = dict[a]

    # for i in range(mdp.num_row):
    #     for j in range(mdp.num_col):
    #         print(pi[i][j])

    return pi
    # ========================


def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
