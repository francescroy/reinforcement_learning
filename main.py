

# Thi is a toy problem modeled as a Markov decision process.
# Author: Francesc Roy Campderrós

import numpy as np
from random import *

X_SIZE =5
Y_SIZE =5
NUM_STATES = X_SIZE * Y_SIZE
GAMMA = 0.90
PREMI_X, PREMI_Y = 2,2
FINAL_STATE = False

class ChanceNode:
    def __init__(self, x,y,action,reward):
        self.action = action
        self.x = x
        self.y = y
        self.trans_probabilities = np.zeros((NUM_STATES,), dtype=float)
        self.reward = reward

        # vaig a especificar 3 trans_probabilities que no seran 0...
        if action=='N':

            self.set_trans_probability(x, y + 1, 0.8)
            self.set_trans_probability(x + 1, y, 0.1)
            self.set_trans_probability(x - 1, y, 0.1)
        elif action=='S':

            self.set_trans_probability(x, y - 1, 0.8)
            self.set_trans_probability(x + 1, y, 0.1)
            self.set_trans_probability(x - 1, y, 0.1)
        elif action=='W':

            self.set_trans_probability(x - 1, y, 0.8)
            self.set_trans_probability(x, y + 1, 0.1)
            self.set_trans_probability(x, y - 1, 0.1)
        else:

            self.set_trans_probability(x + 1, y, 0.8)
            self.set_trans_probability(x, y + 1, 0.1)
            self.set_trans_probability(x, y - 1, 0.1)

    def set_trans_probability(self,x,y,prob):

        if(0 <= x and x <= X_SIZE-1 and 0 <= y and y <= Y_SIZE-1):
            self.trans_probabilities[x + y*Y_SIZE] = prob
        else:
            self.trans_probabilities[self.x + self.y*Y_SIZE] += prob

    def possible_next_sates(self):

        possible_sates = []

        for x in range(X_SIZE):
            for y in range(Y_SIZE):
                prob_to_that_state = self.trans_probabilities[x + y * Y_SIZE]

                if prob_to_that_state != 0:
                    possible_sates.append([find_state(x, y, states), prob_to_that_state])
                    # print (str(x) + " - " +str(y) + " with prob: "+ str(prob_to_that_state))

        return possible_sates

    def next_state(self, states):

        possible_sates = self.possible_next_sates()

        # Com a minim hi haura dos possible_state's no?
        random_int = randint(0, 9)
        definitive_next_state = None
        definitive_reward = None
        #print(random_int)

        if random_int < possible_sates[0][1]*10:
            definitive_next_state=possible_sates[0][0]


        elif random_int < (possible_sates[0][1]+possible_sates[1][1])*10:
            definitive_next_state=possible_sates[1][0]


        else:
            definitive_next_state=possible_sates[2][0]



        return definitive_next_state

class State:
    def __init__(self, x, y, end):
        self.x = x
        self.y = y
        self.end = end
        self.chance_nodes = None

        if end==False:
            self.chance_nodes = [ChanceNode(x,y,'N',compute_reward(x,y,'N')),ChanceNode(x,y,'S',compute_reward(x,y,'S')),ChanceNode(x,y,'W',compute_reward(x,y,'W')),ChanceNode(x,y,'E',compute_reward(x,y,'E'))]

    """
    def paint(self):

        for y in range(Y_SIZE-1, -1, -1):
            for x in range(X_SIZE):

                if (x == self.x and y == self.y):
                    print("X", end=" ")
                else:
                    print("0", end=" ")

            print()
        print()
        """

    def next_state(self, action, states):

        if self.end==True:
            return self
        if action=='N':
            return self.chance_nodes[0].next_state(states)
        if action=='S':
            return self.chance_nodes[1].next_state(states)
        if action=='W':
            return self.chance_nodes[2].next_state(states)
        if action=='E':
            return self.chance_nodes[3].next_state(states)

    def get_chance_node(self, action):
        if self.end ==True:
            return None
        if action == 'N':
            return self.chance_nodes[0]
        if action == 'S':
            return self.chance_nodes[1]
        if action == 'W':
            return self.chance_nodes[2]
        if action == 'E':
            return self.chance_nodes[3]


def find_state(x,y,states):

    result = None

    for s in states:
        if s.x==x and s.y==y:
            result = s

    return result

def compute_reward(x,y,action):

    desired_x =x
    desired_y =y

    if   action=='N':
        desired_y = desired_y + 1
    elif action=='S':
        desired_y = desired_y - 1
    elif action == 'W':
        desired_x = desired_x - 1
    elif action == 'E':
        desired_x = desired_x + 1

    if (0 <= desired_x and desired_x <= X_SIZE - 1 and 0 <= desired_y and desired_y <= Y_SIZE - 1):
        return -pow(pow(desired_x-PREMI_X,2) + pow(desired_y-PREMI_Y,2),0.5)
    else:
        return -pow(pow(x-PREMI_X,2) + pow(y-PREMI_Y,2),0.5)

def sumatori(state,action,Vt_before):

    possible_states = state.get_chance_node(action).possible_next_sates()
    sumatori=0.0

    for possible_state in possible_states:
        sumatori= sumatori + possible_state[1]*Vt_before[possible_state[0].x + possible_state[0].y * Y_SIZE]

    return sumatori

def get_random_policy(states):

    policy= []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if(find_state(x,y,states).end==False):

                policy.append("N")
                """
                random_int = randint(0, 3)
                if random_int==0:
                    policy.append("N")
                elif random_int==1:
                    policy.append("S")
                elif random_int == 2:
                    policy.append("W")
                else:
                    policy.append("E")
                """

            else:
                policy.append(None)

    return policy

def get_max_and_best_action(q_opts):

    best_so_far = q_opts[0]

    if(q_opts[1][0]>best_so_far[0]):
        best_so_far=q_opts[1]
    if(q_opts[2][0]>best_so_far[0]):
        best_so_far=q_opts[2]
    if(q_opts[3][0]>best_so_far[0]):
        best_so_far=q_opts[3]
    return best_so_far


if __name__ == '__main__':

    states = []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if (x==PREMI_X and y == PREMI_Y):
                states.append(State(x,y,FINAL_STATE))
            else:
                states.append(State(x, y, False))


    policy_random_example = get_random_policy(states) # is simply an list of strings...
















    # POLICY EVALUATION
    V=[0.0]*NUM_STATES
    Vt_before = [0.0]*NUM_STATES

    for t in range(10000):
        for i in range(X_SIZE):
            for j in range(Y_SIZE):

                s = find_state(i,j,states)
                action = policy_random_example[i + j * Y_SIZE]

                if s.end==False:
                    V[i + j*Y_SIZE]= s.get_chance_node(action=action).reward + GAMMA*sumatori(s,action,Vt_before) # this q_opt(s,pi(s))
                    Vt_before[i + j * Y_SIZE]= V[i + j*Y_SIZE]


    for i in range(X_SIZE):
        for j in range(Y_SIZE):
            print(str(i) + " " + str(j) +": "+ str(V[i+j*Y_SIZE]))








    # VALUE ITERATION
    V_opt = [0.0] * NUM_STATES
    Vt_opt_before = [0.0] * NUM_STATES
    policy_optimum = [None] * NUM_STATES

    for t in range(10000):
        for i in range(X_SIZE):
            for j in range(Y_SIZE):

                s = find_state(i, j, states)
                if s.end == False:

                    q_otps = []
                    q_otps.append([s.get_chance_node(action='N').reward + GAMMA * sumatori(s, 'N', Vt_before), 'N'])
                    q_otps.append([s.get_chance_node(action='S').reward + GAMMA * sumatori(s, 'S', Vt_before), 'S'])
                    q_otps.append([s.get_chance_node(action='W').reward + GAMMA * sumatori(s, 'W', Vt_before), 'W'])
                    q_otps.append([s.get_chance_node(action='E').reward + GAMMA * sumatori(s, 'E', Vt_before), 'E'])

                    V_opt[i + j * Y_SIZE], policy_optimum[i + j * Y_SIZE] = get_max_and_best_action(q_otps)
                    Vt_opt_before[i + j * Y_SIZE] = V_opt[i + j * Y_SIZE]

    print()
    for y in range(Y_SIZE - 1, -1, -1):
        for x in range(X_SIZE):

            if policy_optimum[x + y * Y_SIZE] != None:
                print(policy_optimum[x + y * Y_SIZE], end=" ")
            else:
                print(" ", end=" ")
        print()
    print()











    # Suposant que no ens donen les probabilitats, com trobes V de una policy pi?: TD learning...  OR el que ja havia fet al primer REPO amb
    # montecarlo simulation pero nomes es pot usar si hi ha final state, en canvi TD learning...

    # After TD learning, Q learning...

    N = [0] * NUM_STATES
    G = [0.0] * NUM_STATES

    initial_state = find_state(0,0,states)

    for t in range(10000):
        #sample episode:


