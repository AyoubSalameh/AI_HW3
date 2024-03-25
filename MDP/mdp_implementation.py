from copy import deepcopy
import numpy as np

#calculate utility for one action, but also tke in mind the probability in reaching others
def calculate_bellman(mdp, U, i, j, action):
    dict = {'UP': 0, 'DOWN': 1, 'RIGHT': 2, 'LEFT': 3}
    sum = 0
    #the probabilitie in taking the current action
    prob = mdp.transition_function[action]
    for a in mdp.actions.keys():
        #getting the next step we reach if we take a
        i_next, j_next = mdp.step((i,j), a)
        #not sure prob[a] works
        sum += prob[dict[a]] * U[i_next][j_next]
    return sum

def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U_tag = deepcopy(U_init)
    U = []
    while True:
        delta = 0
        U = deepcopy(U_tag)
        #TODO: code
        for i in mdp.num_row:
            for j in mdp.num_col:
                if mdp.board[i][j] == 'WALL':
                    continue
            reward = mdp.board[i][j]
            #TODO: make sure the helper function works. currently its a black box
            probabilities = [calculate_bellman(mdp, U, i, j, action) for action in mdp.action.keys()]
            max_util = max(probabilities)
            U_tag[i][j] = reward + mdp.gamma * max_util
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
    raise NotImplementedError
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
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
    raise NotImplementedError
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
