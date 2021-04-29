

# Thi is a toy problem modeled as a Markov decision process.
# Author: Francesc Roy Campderrós

import numpy as np
from random import *
import math

X_SIZE =5
Y_SIZE =5
NUM_STATES = X_SIZE * Y_SIZE
GAMMA = 0.90
PREMI_X, PREMI_Y = 2,2
FINAL_STATE = True # Can be false if using TD-learning or Policy evaluation...
COST_STEP = 0.10

class ChanceNode:
    def __init__(self, x,y,action,reward):
        self.action = action
        self.x = x
        self.y = y
        self.trans_probabilities = np.zeros((NUM_STATES,), dtype=float)
        self.reward = reward

        if action=='N':

            self.set_trans_probability(x, y + 1, 0.7)
            self.set_trans_probability(x + 1, y, 0.15)
            self.set_trans_probability(x - 1, y, 0.15)
        elif action=='S':

            self.set_trans_probability(x, y - 1, 0.7)
            self.set_trans_probability(x + 1, y, 0.15)
            self.set_trans_probability(x - 1, y, 0.15)
        elif action=='W':

            self.set_trans_probability(x - 1, y, 0.7)
            self.set_trans_probability(x, y + 1, 0.15)
            self.set_trans_probability(x, y - 1, 0.15)
        elif action=='E':

            self.set_trans_probability(x + 1, y, 0.7)
            self.set_trans_probability(x, y + 1, 0.15)
            self.set_trans_probability(x, y - 1, 0.15)
        elif action=='·':
            self.set_trans_probability(x ,y, 0.8)
            self.set_trans_probability(x, y + 1, 0.05)
            self.set_trans_probability(x, y - 1, 0.05)
            self.set_trans_probability(x + 1, y, 0.05)
            self.set_trans_probability(x - 1, y, 0.05)

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
        random_int = randint(0, 99)
        definitive_next_state = None

        acumulative=0

        for s in possible_sates:

            acumulative = acumulative + s[1]

            if random_int < acumulative * 100:
                definitive_next_state = s[0]
                break


        return definitive_next_state

class State:
    def __init__(self, x, y, end):
        self.x = x
        self.y = y
        self.end = end
        self.chance_nodes = None

        if end==False:
            self.chance_nodes = [ChanceNode(x,y,'N',compute_reward(x,y,'N')),ChanceNode(x,y,'S',compute_reward(x,y,'S')),ChanceNode(x,y,'W',compute_reward(x,y,'W')),ChanceNode(x,y,'E',compute_reward(x,y,'E')),ChanceNode(x,y,'·',compute_reward(x,y,'·'))]

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
        if action=='·':
            return self.chance_nodes[4].next_state(states)

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
        if action == '·':
            return self.chance_nodes[4]


def find_state(x,y,states):

    result = None

    for s in states:
        if s.x==x and s.y==y:
            result = s

    return result

def compute_cost(x,y):
    return (pow(x - PREMI_X, 2) + pow(y - PREMI_Y, 2))

def compute_reward(x,y,action):

    desired_x =x
    desired_y =y

    if action=='N':
        desired_y = desired_y + 1
    elif action=='S':
        desired_y = desired_y - 1
    elif action == 'W':
        desired_x = desired_x - 1
    elif action == 'E':
        desired_x = desired_x + 1
    elif action == '·':
        pass

    ## Here

    if (0 <= desired_x and desired_x <= X_SIZE - 1 and 0 <= desired_y and desired_y <= Y_SIZE - 1):
        if action!='·':
            return (compute_cost(x,y) - compute_cost(desired_x,desired_y)) - COST_STEP
        else:
            return 0.0
    else:
        return -COST_STEP

def get_random_policy(states):

    policy= []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if(find_state(x,y,states).end==False):

                if y>math.floor(Y_SIZE/2.0):
                    policy.append("S")
                else:
                    policy.append("N")

            else:
                policy.append(None)

    return policy

def list_contains(states, state): #returns true/false and an array of positions...

    found= False
    positions = []

    counter =0
    for s in states:
        if s.x==state.x and s.y == state.y:
            found= True
            positions.append(counter)

        counter = counter +1

    return found,positions

def compute_reward_of_episode(states_episode,policy):

    reward=0.0
    number_of_steps=0

    for s in states_episode:
        if s.end==False:
            reward = reward + s.get_chance_node(policy[s.x + s.y * Y_SIZE]).reward * pow(GAMMA, number_of_steps)
            number_of_steps = number_of_steps +1
    return reward

def sumatori(state,action,Vt_before):

    possible_states = state.get_chance_node(action).possible_next_sates()
    sumatori=0.0

    for possible_state in possible_states:
        sumatori= sumatori + possible_state[1]*Vt_before[possible_state[0].x + possible_state[0].y * Y_SIZE]

    return sumatori

def get_max_and_best_action(q_opts):

    best_so_far = q_opts[0]

    if(q_opts[1][0] > best_so_far[0]):
        best_so_far = q_opts[1]
    if(q_opts[2][0] > best_so_far[0]):
        best_so_far = q_opts[2]
    if(q_opts[3][0] > best_so_far[0]):
        best_so_far = q_opts[3]
    if(q_opts[4][0] > best_so_far[0]):
        best_so_far = q_opts[4]

    return best_so_far





if __name__ == '__main__':

    states = []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if (x==PREMI_X and y == PREMI_Y):
                states.append(State(x,y,FINAL_STATE))
            else:
                states.append(State(x, y, False))


    policy_random_example = get_random_policy(states) # is simply a list of strings...

    print("What do you want to do?:")

    option_selected = input()

    if option_selected == "1":

        # POLICY EVALUATION
        V=[0.0]*NUM_STATES
        Vt_before = [0.0]*NUM_STATES

        for t in range(10000):
            for i in range(X_SIZE):
                for j in range(Y_SIZE):

                    s = find_state(i,j,states)
                    action = policy_random_example[i + j * Y_SIZE]

                    if s.end==False:

                        q_opt = s.get_chance_node(action=action).reward + GAMMA*sumatori(s,action,Vt_before)

                        V[i + j*Y_SIZE]=  q_opt
                        Vt_before[i + j * Y_SIZE]= V[i + j*Y_SIZE]


        for i in range(X_SIZE):
            for j in range(Y_SIZE):
                print(str(i) + " " + str(j) +": "+ str(V[i+j*Y_SIZE]))

    if option_selected == "2":

        # VALUE ITERATION
        V_opt = [0.0] * NUM_STATES
        Vt_opt_before = [0.0] * NUM_STATES
        policy_optimum = [None] * NUM_STATES

        for t in range(10000):

            if t%(10000/10)==0:
                print("|")

            for i in range(X_SIZE):
                for j in range(Y_SIZE):

                    s = find_state(i, j, states)
                    if s.end == False:

                        q_otps = []
                        q_otps.append([s.get_chance_node(action='N').reward + GAMMA * sumatori(s, 'N', Vt_opt_before), 'N'])
                        q_otps.append([s.get_chance_node(action='S').reward + GAMMA * sumatori(s, 'S', Vt_opt_before), 'S'])
                        q_otps.append([s.get_chance_node(action='W').reward + GAMMA * sumatori(s, 'W', Vt_opt_before), 'W'])
                        q_otps.append([s.get_chance_node(action='E').reward + GAMMA * sumatori(s, 'E', Vt_opt_before), 'E'])
                        q_otps.append([s.get_chance_node(action='·').reward + GAMMA * sumatori(s, '·', Vt_opt_before), '·'])

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

    if option_selected == "1":

        ############################################
        ############################################
        ##### SHOULD EXIST AND END STATE ###########
        ############################################
        ############################################

        N = [0] * NUM_STATES
        G = [0.0] * NUM_STATES
        V_monte_carlo = [0.0] * NUM_STATES

        initial_state = find_state(0,0,states)

        for loop in range(100000):

            if(loop%(100000/10)==0):
                print("|")

            states_episode = []
            states_episode.append(initial_state)
            number_of_states = 1

            while (states_episode[number_of_states-1].end==False):

                actual_state = states_episode[number_of_states-1]
                next_state = actual_state.next_state(policy_random_example[actual_state.x + actual_state.y * Y_SIZE],states)

                states_episode.append(next_state)
                number_of_states = number_of_states +1
                #print(str(actual_state.x)+ " " + str(actual_state.y))


            G_t = [] # sera tant llarga com states_episode

            for t in range(number_of_states):
                reward_from_t = compute_reward_of_episode(states_episode[t::],policy_random_example)
                G_t.append(reward_from_t)

            for x in range(X_SIZE):
                for y in range(Y_SIZE):
                    state = find_state(x,y,states)
                    cont,pos = list_contains(states_episode, state) #returns true/false and an array of positions...
                    if cont==True:
                        N[x + y*Y_SIZE] = N[x + y*Y_SIZE] +1
                        G[x + y*Y_SIZE] = G[x + y*Y_SIZE] +G_t[pos[0]]
                        V_monte_carlo[x + y*Y_SIZE] = G[x + y*Y_SIZE] / N[x + y*Y_SIZE]

        for i in range(X_SIZE):
            for j in range(Y_SIZE):
                print(str(i) + " " + str(j) + ": " + str(V_monte_carlo[i + j * Y_SIZE]))




    # Suposant que no ens donen les probabilitats, com trobes V de una policy pi?: TD learning...  OR el que ja havia fet al primer REPO amb
    # montecarlo simulation pero nomes es pot usar si hi ha final state, en canvi TD learning...

    # Osigui model free es com RL ja no...? CLAU...

    # After TD learning, Q learning...

    # potser podria usar threads per usar diferents CPU's...
    # es endavant -> imagina que les transition probabilities cambiessin through time... que en el fons es el que
    # passa al autoscaling problem...

    # l'ultim montecarlo esta bé perque aconsegueix estimator of V que es unbiased!! tot i que es veu que la variance es bastant gran...