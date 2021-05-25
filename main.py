

# This is a toy problem modeled as a Markov decision process.
# Author: Francesc Roy Campderrós

import numpy as np
from random import *
import math
import sys

X_SIZE =7 # 5,7,9
Y_SIZE =7 # 5,7,9
NUM_STATES = X_SIZE * Y_SIZE
GAMMA = 0.90
OPTIMAL_X, OPTIMAL_Y = 3,3 # 2,3,4
OPTIMAL_FINAL_STATE = False # Can be false if using TD-learning or DP methods but must be true if some MonteCarlo method...
COST_STEP = 0.10
NUM_ACTIONS = 5

INCREMENTAL_MONTECARLO = False
EVERY_VISIT_MONTECARLO = False # If INCREMENTAL_MONTECARLO is True then it doesn't matter...

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

    def possible_next_sates(self,states):

        possible_sates = []

        for x in range(X_SIZE):
            for y in range(Y_SIZE):
                prob_to_that_state = self.trans_probabilities[x + y * Y_SIZE]

                if prob_to_that_state != 0:
                    possible_sates.append([find_state(x, y, states), prob_to_that_state])
                    # print (str(x) + " - " +str(y) + " with prob: "+ str(prob_to_that_state))

        return possible_sates

    def next_state(self, states):

        possible_sates = self.possible_next_sates(states)

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

    def next_state(self, action, states):

        if self.end==True:
            return self # o None?
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

    reward = 0.0

    if (0 <= desired_x and desired_x <= X_SIZE - 1 and 0 <= desired_y and desired_y <= Y_SIZE - 1):
        if action!='·':
            reward =  (compute_cost(x,y) - compute_cost(desired_x,desired_y)) - COST_STEP
    else:
        reward = -COST_STEP


    return reward

def get_nice_policy(states):
    policy = [None]*NUM_STATES

    for x in range(0,int((X_SIZE/2))):
        for y in range(Y_SIZE):
            policy[x + y*Y_SIZE] = 'S'

    for x in range(X_SIZE):
        for y in range(0,int((Y_SIZE/2))):
            policy[x + y * Y_SIZE] = 'E'

    for x in range(int((X_SIZE/2))+1,X_SIZE):
        for y in range(Y_SIZE):
            policy[x + y*Y_SIZE] = 'N'

    for x in range(int((X_SIZE/2)),X_SIZE):
        for y in range(int((Y_SIZE/2)+1),Y_SIZE):
            policy[x + y * Y_SIZE] = 'W'

    for x in range(X_SIZE):
        for y in range(Y_SIZE):
            if policy[x + y * Y_SIZE] == None:
                if states[x + y * Y_SIZE].end==False:
                    policy[x + y * Y_SIZE] = '·'



    return policy

def get_random_policy(states):

    policy= []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if(find_state(x,y,states).end==False):


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

def sumatori(state,action,Vt_before,states):

    possible_states = state.get_chance_node(action).possible_next_sates(states)
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

def print_policy(policy):
    print()
    for y in range(Y_SIZE - 1, -1, -1):
        for x in range(X_SIZE):

            if policy[x + y * Y_SIZE] != None:
                print(policy[x + y * Y_SIZE], end=" ")
            else:
                print(" ", end=" ")
        print()
    print()

def get_random_state(states):
    random_state = find_state(randint(0, X_SIZE - 1), randint(0, Y_SIZE - 1), states)
    while random_state.end == True:
        random_state = find_state(randint(0, X_SIZE - 1), randint(0, Y_SIZE - 1), states)
    return random_state

def generate_episode(initial_state,policy,states):

    states_episode = []

    states_episode.append(initial_state)
    current_state = initial_state

    while (current_state.end == False):
        next_state = current_state.next_state(policy[current_state.x + current_state.y * Y_SIZE], states)
        states_episode.append(next_state)
        current_state = next_state


    return states_episode,len(states_episode)

def print_wait_info(loop,number_of_iterations):
    if (loop % (number_of_iterations / 10) == 0 and loop != 0):
        print("|", end=' ')
        sys.stdout.flush()

def get_num_action(action):
    if(action=='N'):
        return 0
    if(action=='S'):
        return 1
    if(action=='W'):
        return 2
    if(action=='E'):
        return 3
    if(action=='·'):
        return 4

def get_action(action_num):
    if(action_num==0):
        return 'N'
    if(action_num==1):
        return 'S'
    if(action_num==2):
        return 'W'
    if(action_num==3):
        return 'E'
    if(action_num==4):
        return '·'

def argmax_a(Q,state):

    best_action = get_action(0)
    best_q = Q[0][state.x + state.y * Y_SIZE]

    if Q[1][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(1)
        best_q = Q[1][state.x + state.y * Y_SIZE]

    if Q[2][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(2)
        best_q = Q[2][state.x + state.y * Y_SIZE]

    if Q[3][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(3)
        best_q = Q[3][state.x + state.y * Y_SIZE]

    if Q[4][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(4)
        best_q = Q[4][state.x + state.y * Y_SIZE]

    return best_action

def max_a(Q,state):

    best_action = get_action(0)
    best_q = Q[0][state.x + state.y * Y_SIZE]

    if Q[1][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(1)
        best_q = Q[1][state.x + state.y * Y_SIZE]

    if Q[2][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(2)
        best_q = Q[2][state.x + state.y * Y_SIZE]

    if Q[3][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(3)
        best_q = Q[3][state.x + state.y * Y_SIZE]

    if Q[4][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(4)
        best_q = Q[4][state.x + state.y * Y_SIZE]

    return best_q




def main():

    states = []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if (x==OPTIMAL_X and y == OPTIMAL_Y):
                states.append(State(x,y,OPTIMAL_FINAL_STATE))
            else:
                states.append(State(x, y, False))


    # policy_example = get_nice_policy(states) # is simply a list of strings...
    policy_example = get_random_policy(states)  # is simply a list of strings...

    print_policy(policy_example)

    print("What do you want to do?:")

    V_to_compare = []

    option_selected = input()




    # MODEL-BASED MDP ALGO: POLICY EVALUATION, GIVEN A POLICY, WHAT IS THE EXPECTED REWARD?
    if option_selected == "1":

        # POLICY EVALUATION
        V=[0.0]*NUM_STATES
        Vt_before = [0.0]*NUM_STATES

        number_of_iterations = 10000

        for t in range(number_of_iterations):
            for i in range(X_SIZE):
                for j in range(Y_SIZE):

                    s = find_state(i,j,states)
                    action = policy_example[i + j * Y_SIZE]

                    if s.end==False:

                        V[i + j*Y_SIZE]=  s.get_chance_node(action=action).reward + GAMMA*sumatori(s,action,Vt_before,states)
                        Vt_before[i + j * Y_SIZE]= V[i + j*Y_SIZE]

        print_V(V)
        V_to_compare = V

    # MODEL-BASED MDP ALGO: VALUE ITERATION, FIND OPTIMAL POLICY
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

                        possible_Vs = []
                        possible_Vs.append([s.get_chance_node(action='N').reward + GAMMA * sumatori(s, 'N', Vt_opt_before,states), 'N'])
                        possible_Vs.append([s.get_chance_node(action='S').reward + GAMMA * sumatori(s, 'S', Vt_opt_before,states), 'S'])
                        possible_Vs.append([s.get_chance_node(action='W').reward + GAMMA * sumatori(s, 'W', Vt_opt_before,states), 'W'])
                        possible_Vs.append([s.get_chance_node(action='E').reward + GAMMA * sumatori(s, 'E', Vt_opt_before,states), 'E'])
                        possible_Vs.append([s.get_chance_node(action='·').reward + GAMMA * sumatori(s, '·', Vt_opt_before,states), '·'])

                        V_opt[i + j * Y_SIZE], policy_optimum[i + j * Y_SIZE] = get_max_and_best_action(possible_Vs)
                        Vt_opt_before[i + j * Y_SIZE] = V_opt[i + j * Y_SIZE]

        print_policy(policy_optimum)

    # MODEL-FREE MDP ALGO: MC POLICY EVALUATION, GIVEN A POLICY, WHAT IS THE [ESTIMATED] EXPECTED REWARD?
    if option_selected == "3":

        ############################################
        ############################################
        ##### SHOULD EXIST AND END STATE ###########
        ############################################
        ############################################

        # An also an important thing I think is that you can not get trapped in a loop.

        N = [0] * NUM_STATES
        G = [0.0] * NUM_STATES
        V_monte_carlo = [0.0] * NUM_STATES

        number_of_iterations = 1000000

        for loop in range(number_of_iterations): # Each loop is an episode.

            print_wait_info(loop,number_of_iterations)

            states_episode,number_of_states = generate_episode(get_random_state(states),policy_example,states)

            G_t = [] # sera tant llarga com states_episode
            for t in range(number_of_states):
                reward_from_t = compute_reward_of_episode(states_episode[t::],policy_example)
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
            #else:
            #    for t in range(number_of_states):
            #        state = states_episode[t]
            #        N[state.x + state.y * Y_SIZE] = N[state.x + state.y * Y_SIZE] + 1
            #        ALPHA = 1/N[state.x + state.y * Y_SIZE] # Equal to EVERY_VISIT_MONTECARLO...
            #        V_monte_carlo[state.x + state.y * Y_SIZE] = V_monte_carlo[state.x + state.y * Y_SIZE] + ALPHA*(G_t[t] - V_monte_carlo[state.x + state.y * Y_SIZE])

        print_V(V_monte_carlo)
        print(sum(list(abs(np.array(V_to_compare) - np.array(V_monte_carlo)))))

    # MODEL-FREE MDP ALGO: TD LEARNING, GIVEN A POLICY, WHAT IS THE [ESTIMATED] EXPECTED REWARD?
    if option_selected == "1":
        # Let's develop TD learning! El problema es que has de tenir una policy que es mogui per toooots els estats
        # Clar perque fer un rand(),rand() a current_state per anar saltant d'estats es com fer trampa...

        # La gracia d'aquest respecte l'anterior és que pot tractar MDP's sense end state, i si fas tants moviments com al montecarlo
        # aconseguiras una precisió molt semblant...

        number_of_iterations = 10000000
        V = [0.0] * NUM_STATES
        ALPHA = 0.001  # Which is the right value? After 50% of iteration decay... after 80% decay... LEARNING RATE...
        current_state = get_random_state(states)

        for t in range(number_of_iterations): # Each t is a movement (s, a, r, s)

            print_wait_info(t,number_of_iterations)

            action = policy_example[current_state.x + current_state.y * Y_SIZE]
            reward = current_state.get_chance_node(action).reward
            next_state = current_state.next_state(action,states)

            V_current = V[current_state.x + current_state.y * Y_SIZE]
            V_next = V[next_state.x + next_state.y * Y_SIZE]

            if t%(number_of_iterations/10)== 0 and t!=0:
                ALPHA= ALPHA/2

            V[current_state.x + current_state.y * Y_SIZE] = V_current + ALPHA * ((reward+GAMMA*V_next)-V_current)

            if next_state.end == False:
                current_state = next_state
            else:
                current_state = get_random_state(states)


        print_V(V)
        print(sum(list(abs(np.array(V_to_compare) - np.array(V)))))

    # MODEL-FREE MDP ALGO: MC CONTROL, FIND [ESTIMATED] OPTIMAL POLICY
    if option_selected == "4":

        ############################################
        ############################################
        ##### SHOULD EXIST AN END STATE ############
        ############################################
        ############################################

        EPSILON = 0.05

        policy_actual = policy_example
        policy_improved = [None] * NUM_STATES

        # Policy evaluation:

        N = []
        G = []
        Q = []

        for i in range(NUM_ACTIONS):
            N.append([0] * NUM_STATES)
            G.append([0.0] * NUM_STATES)
            Q.append([0.0] * NUM_STATES)

        number_of_iterations = 10000

        for number_of_improvements in range(100000):

            for loop in range(number_of_iterations): # Each loop is an episode.

                print_wait_info(loop,number_of_iterations)
                states_episode,number_of_states = generate_episode(get_random_state(states),policy_actual,states)

                G_t = [] # sera tant llarga com states_episode
                for t in range(number_of_states):
                    reward_from_t = compute_reward_of_episode(states_episode[t::],policy_actual)
                    G_t.append(reward_from_t)

                for x in range(X_SIZE):
                    for y in range(Y_SIZE):
                        state = find_state(x, y, states)
                        if state.end==False:
                            num_action = get_num_action(policy_actual[state.x + state.y * Y_SIZE])
                            cont, pos = list_contains(states_episode, state)  # returns true/false and an array of positions...
                            if cont == True:
                                if EVERY_VISIT_MONTECARLO == False:
                                    N[num_action][x + y * Y_SIZE] = N[num_action][x + y * Y_SIZE] + 1
                                    G[num_action][x + y * Y_SIZE] = G[num_action][x + y * Y_SIZE] + G_t[pos[0]]
                                    Q[num_action][x + y * Y_SIZE] = G[num_action][x + y * Y_SIZE] / N[num_action][x + y * Y_SIZE]
                                else:
                                    for p in pos:
                                        N[num_action][x + y * Y_SIZE] = N[num_action][x + y * Y_SIZE] + 1
                                        G[num_action][x + y * Y_SIZE] = G[num_action][x + y * Y_SIZE] + G_t[p]
                                        Q[num_action][x + y * Y_SIZE] = G[num_action][x + y * Y_SIZE] / N[num_action][x + y * Y_SIZE]


            # Policy improvement:
            for x in range(X_SIZE):
                for y in range(Y_SIZE):

                    state = find_state(x, y, states)

                    if (state.end==False):

                        policy_improved[x + y * Y_SIZE] = argmax_a(Q,state)

                        random_int = randint(0, 99)
                        if random_int<int(EPSILON*100):
                            policy_improved[x + y * Y_SIZE] = get_action(randint(0, 4))

            print_policy(policy_improved)
            print(number_of_improvements)
            policy_actual=policy_improved

    # MODEL-FREE MDP ALGO: Q-LEARNING, FIND [ESTIMATED] OPTIMAL POLICY
    if option_selected == "5":

        policy_actual = policy_example

        ALPHA = 0.001  # Which is the right value? After 50% of iteration decay... after 80% decay... LEARNING RATE...
        current_state = get_random_state(states)
        Q = []
        for i in range(NUM_ACTIONS):
            Q.append([0.0] * NUM_STATES)

        EPSILON = 1.00
        number_of_iterations = 10000000        
        DECAYING_EPSILON = 1.0/number_of_iterations

        for t in range(number_of_iterations):

            print_wait_info(t, number_of_iterations)

            action = policy_actual[current_state.x + current_state.y * Y_SIZE]
            reward = current_state.get_chance_node(action).reward
            next_state = current_state.next_state(action, states)

            Q_t_minus_1 = Q[get_num_action(action)][current_state.x + current_state.y * Y_SIZE]

            if t%(number_of_iterations/10)== 0 and t!=0:
                ALPHA= ALPHA/2

            Q[get_num_action(action)][current_state.x + current_state.y *Y_SIZE] = Q_t_minus_1 + ALPHA * (reward + GAMMA * max_a(Q, next_state) - Q_t_minus_1)

            policy_actual[current_state.x + current_state.y * Y_SIZE] = argmax_a(Q, current_state)
            random_int = randint(0, 99)
            if random_int < int(EPSILON * 100):
                policy_actual[current_state.x + current_state.y * Y_SIZE] = get_action(randint(0, 4))

            EPSILON= EPSILON - DECAYING_EPSILON

            current_state = next_state

        print_policy(policy_actual)









    # Suposant que no ens donen les probabilitats, com trobes V de una policy pi?: TD learning... Osigui model free es com RL ja no...? CLAU...

    # After TD learning, Q learning...

    # potser podria usar threads per usar diferents CPU's...
    # mes endavant -> imagina que les transition probabilities cambiessin through time... que en el fons es el que passa al autoscaling problem...

    # l'ultim montecarlo esta bé perque aconsegueix estimator of V que es unbiased!! tot i que es veu que la variance es bastant gran...

    ## THEORY ##
    # It would be nice to undestand why DP policy evaluation algo. works... is because you are using bootstrapping...
    # It's much easier to understand MC policy evaluation algo. works... it relies on sampling, not on bootstrapping...

if __name__ == '__main__':
    main()