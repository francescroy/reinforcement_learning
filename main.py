

# Thi is a toy problem modeled as a Markov decision process.
# Author: Francesc Roy Campderrós

import numpy as np
from random import *
import math
import sys

X_SIZE =5
Y_SIZE =5
NUM_STATES = X_SIZE * Y_SIZE
GAMMA = 0.90
OPTIMAL_X, OPTIMAL_Y = 2,2
OPTIMAL_FINAL_STATE = False # Can be false if using TD-learning or DP methods but must be true if some MonteCarlo method for policy evaluation...
COST_STEP = 0.10

INCREMENTAL_MONTECARLO = True
EVERY_VISIT_MONTECARLO = True # If INCREMENTAL_MONTECARLO is True then it doesn't matter...


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
    return (pow(x - OPTIMAL_X, 2) + pow(y - OPTIMAL_Y, 2))

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

                """
                rand_dir = randint(0, 4)
                if rand_dir==0:
                    policy.append('N')
                if rand_dir==1:
                    policy.append('S')
                if rand_dir==2:
                    policy.append('W')
                if rand_dir==3:
                    policy.append('E')
                if rand_dir==4:
                    policy.append('·')
                
                """

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

def print_V(V):
    for i in range(X_SIZE):
        for j in range(Y_SIZE):
            print(str(i) + " " + str(j) + ": " + str(V[i + j * Y_SIZE]))



if __name__ == '__main__':

    states = []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if (x==OPTIMAL_X and y == OPTIMAL_Y):
                states.append(State(x,y,OPTIMAL_FINAL_STATE))
            else:
                states.append(State(x, y, False))

    policy_random_example = get_random_policy(states) # is simply a list of strings...

    print("What do you want to do?:")

    V_to_compare = []

    option_selected = input()

    if option_selected == "1":

        # POLICY EVALUATION
        V=[0.0]*NUM_STATES
        Vt_before = [0.0]*NUM_STATES

        number_of_iterations = 10000

        for t in range(number_of_iterations):
            for i in range(X_SIZE):
                for j in range(Y_SIZE):

                    s = find_state(i,j,states)
                    action = policy_random_example[i + j * Y_SIZE]

                    if s.end==False:

                        q_opt = s.get_chance_node(action=action).reward + GAMMA*sumatori(s,action,Vt_before)

                        V[i + j*Y_SIZE]=  q_opt
                        Vt_before[i + j * Y_SIZE]= V[i + j*Y_SIZE]

        print_V(V)
        V_to_compare = V

    if option_selected == "2":

        # VALUE ITERATION
        V_opt = [0.0] * NUM_STATES
        Vt_opt_before = [0.0] * NUM_STATES
        policy_optimum = [None] * NUM_STATES

        number_of_iterations= 10000

        for t in range(number_of_iterations):

            if (t%(number_of_iterations/10)==0 and t!=0):
                print("|", end=' ')
                sys.stdout.flush()

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

    if option_selected == "3":

        ############################################
        ############################################
        ##### SHOULD EXIST AND END STATE ###########
        ############################################
        ############################################

        N = [0] * NUM_STATES
        G = [0.0] * NUM_STATES
        V_monte_carlo = [0.0] * NUM_STATES

        initial_state = find_state(0,0,states)

        number_of_iterations=  100000

        for loop in range(number_of_iterations): # Each loop is an episode.

            if(loop%(number_of_iterations/10)==0 and loop!=0):
                print("|" , end = ' ')
                sys.stdout.flush()

            states_episode = []
            states_episode.append(initial_state)
            number_of_states = 1

            while (states_episode[number_of_states-1].end==False):

                current_state = states_episode[number_of_states-1]
                next_state = current_state.next_state(policy_random_example[current_state.x + current_state.y * Y_SIZE],states)

                states_episode.append(next_state)
                number_of_states = number_of_states +1
                #print(str(current_state.x)+ " " + str(current_state.y))


            G_t = [] # sera tant llarga com states_episode

            for t in range(number_of_states):
                reward_from_t = compute_reward_of_episode(states_episode[t::],policy_random_example)
                G_t.append(reward_from_t)

            if INCREMENTAL_MONTECARLO == False:

                for x in range(X_SIZE):
                    for y in range(Y_SIZE):
                        state = find_state(x,y,states)
                        cont,pos = list_contains(states_episode, state) #returns true/false and an array of positions...
                        if cont==True:
                            if EVERY_VISIT_MONTECARLO==False:
                                N[x + y*Y_SIZE] = N[x + y*Y_SIZE] +1
                                G[x + y*Y_SIZE] = G[x + y*Y_SIZE] +G_t[pos[0]]
                                V_monte_carlo[x + y*Y_SIZE] = G[x + y*Y_SIZE] / N[x + y*Y_SIZE]
                            else:
                                for p in pos:
                                    N[x + y * Y_SIZE] = N[x + y * Y_SIZE] + 1
                                    G[x + y * Y_SIZE] = G[x + y * Y_SIZE] + G_t[p]
                                    V_monte_carlo[x + y * Y_SIZE] = G[x + y * Y_SIZE] / N[x + y * Y_SIZE]
            else:
                for t in range(number_of_states):
                    state = states_episode[t]
                    N[state.x + state.y * Y_SIZE] = N[state.x + state.y * Y_SIZE] + 1
                    ALPHA = 1/N[state.x + state.y * Y_SIZE] # Equal to EVERY_VISIT_MONTECARLO...
                    V_monte_carlo[state.x + state.y * Y_SIZE] = V_monte_carlo[state.x + state.y * Y_SIZE] + ALPHA*(G_t[t] - V_monte_carlo[state.x + state.y * Y_SIZE])

        print_V(V_monte_carlo)

    if option_selected == "1":
        # Let's develop TD learning!

        number_of_iterations = 20000000
        V = [0.0] * NUM_STATES
        ALPHA = 0.001  # Which is the right value? After 50% of iteration decay... after 80% decay...

        for t in range(number_of_iterations):

            if (t % (number_of_iterations / 10) == 0 and t != 0):
                print("|", end=' ')
                sys.stdout.flush()

            current_state = find_state(randint(0, X_SIZE - 1), randint(0, Y_SIZE - 1), states)
            action = policy_random_example[current_state.x + current_state.y * Y_SIZE]
            reward = current_state.get_chance_node(action).reward
            next_state = current_state.next_state(action,states)

            V_current = V[current_state.x + current_state.y * Y_SIZE]
            V_next = V[next_state.x + next_state.y * Y_SIZE]

            if t== number_of_iterations/2:
                ALPHA= ALPHA/10
            if t== (number_of_iterations/2 + number_of_iterations/4):
                ALPHA= ALPHA/2

            V[current_state.x + current_state.y * Y_SIZE] = V_current + ALPHA * ((reward+GAMMA*V_next)-V_current)



        #print_V(V)
        print(sum(list(abs(np.array(V_to_compare) - np.array(V)))))

    if option_selected == "4":
        # Let's develop Policy iteration!
        # todo


        print()

    if option_selected == "5":
        # Montecarlo method to find best policy
        # todo


        print()





    # Suposant que no ens donen les probabilitats, com trobes V de una policy pi?: TD learning... Osigui model free es com RL ja no...? CLAU...

    # After TD learning, Q learning...

    # potser podria usar threads per usar diferents CPU's...
    # es endavant -> imagina que les transition probabilities cambiessin through time... que en el fons es el que passa al autoscaling problem...

    # l'ultim montecarlo esta bé perque aconsegueix estimator of V que es unbiased!! tot i que es veu que la variance es bastant gran...





    ## THEROY ##
    # It would be nice to undestand why DP policy evaluation algo. works... is because you are using bootstrapping...
    # It's much easier to understand MC policy evaluation algo. works... it relies on sampling, not on bootstrapping...

