

# Thi is a toy problem modeled as a Markov decision process.
# Author: Francesc Roy Campderrós

import numpy as np
from random import *

X_SIZE =5
Y_SIZE =5
NUM_STATES = X_SIZE * Y_SIZE
NEGATIVE_REWARD_STEP = -0.05
GAMMA = 0.90

class ChanceNode:
    def __init__(self, x,y,action):
        self.action = action
        self.x = x
        self.y = y
        self.trans_probabilities = np.zeros((NUM_STATES,), dtype=float)
        self.rewards = [None] * NUM_STATES

        # vaig a especificar 3 trans_probabilities que no seran 0...
        if action=='N':

            self.set_trans_probability_and_reward(x, y + 1, 0.8,NEGATIVE_REWARD_STEP)
            self.set_trans_probability_and_reward(x + 1, y, 0.1,NEGATIVE_REWARD_STEP)
            self.set_trans_probability_and_reward(x - 1, y, 0.1,NEGATIVE_REWARD_STEP)
        elif action=='S':

            self.set_trans_probability_and_reward(x, y - 1, 0.8,NEGATIVE_REWARD_STEP)
            self.set_trans_probability_and_reward(x + 1, y, 0.1,NEGATIVE_REWARD_STEP)
            self.set_trans_probability_and_reward(x - 1, y, 0.1,NEGATIVE_REWARD_STEP)
        elif action=='W':

            self.set_trans_probability_and_reward(x - 1, y, 0.8,NEGATIVE_REWARD_STEP)
            self.set_trans_probability_and_reward(x, y + 1, 0.1,NEGATIVE_REWARD_STEP)
            self.set_trans_probability_and_reward(x, y - 1, 0.1,NEGATIVE_REWARD_STEP)
        else:

            self.set_trans_probability_and_reward(x + 1, y, 0.8,NEGATIVE_REWARD_STEP)
            self.set_trans_probability_and_reward(x, y + 1, 0.1,NEGATIVE_REWARD_STEP)
            self.set_trans_probability_and_reward(x, y - 1, 0.1,NEGATIVE_REWARD_STEP)

    def set_trans_probability_and_reward(self,x,y,prob,reward):

        #imaginant que no hi ha cas raro de moment...
        if(0 <= x and x <= X_SIZE-1 and 0 <= y and y <= Y_SIZE-1):
            self.trans_probabilities[x + y*Y_SIZE] = prob
            self.rewards[x + y * Y_SIZE] = reward
        else:
            self.trans_probabilities[self.x + self.y*Y_SIZE] += prob
            self.rewards[self.x + self.y*Y_SIZE] = reward

    def possible_next_sates(self):

        possible_sates = []

        for x in range(X_SIZE):
            for y in range(Y_SIZE):
                prob_to_that_state = self.trans_probabilities[x + y * Y_SIZE]
                reward_to_that_state = self.rewards[x + y * Y_SIZE]
                if prob_to_that_state != 0:
                    possible_sates.append([find_state(x, y, states), prob_to_that_state, reward_to_that_state])
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
            definitive_reward=possible_sates[0][2]

        elif random_int < (possible_sates[0][1]+possible_sates[1][1])*10:
            definitive_next_state=possible_sates[1][0]
            definitive_reward=possible_sates[1][2]

        else:
            definitive_next_state=possible_sates[2][0]
            definitive_reward=possible_sates[2][2]


        return definitive_next_state, definitive_reward


class State:
    def __init__(self, x, y, end):
        self.x = x
        self.y = y
        self.end = end
        self.chance_nodes = None

        if end==False:
            self.chance_nodes = [ChanceNode(x,y,'N'),ChanceNode(x,y,'S'),ChanceNode(x,y,'W'),ChanceNode(x,y,'E')]

    def paint(self):

        for y in range(Y_SIZE-1, -1, -1):
            for x in range(X_SIZE):

                if (x == self.x and y == self.y):
                    print("X", end=" ")
                else:
                    print("0", end=" ")

            print()
        print()

    def next_state(self, action, states):

        if self.end==True:
            return self, NEGATIVE_REWARD_STEP
        if action=='N':
            return self.chance_nodes[0].next_state(states)
        if action=='S':
            return self.chance_nodes[1].next_state(states)
        if action=='W':
            return self.chance_nodes[2].next_state(states)
        if action=='E':
            return self.chance_nodes[3].next_state(states)

    def set_reward(self, reward, states):

        neighbour_N = find_state(self.x,self.y+1,states)
        neighbour_S = find_state(self.x,self.y-1,states)
        neighbour_W = find_state(self.x-1,self.y,states)
        neighbour_E = find_state(self.x+1,self.y,states)

        if neighbour_N is not None and neighbour_N.end==False:
            neighbour_N.chance_nodes[1].rewards[self.x + self.y*Y_SIZE] = reward
            neighbour_N.chance_nodes[2].rewards[self.x + self.y*Y_SIZE] = reward
            neighbour_N.chance_nodes[3].rewards[self.x + self.y*Y_SIZE] = reward
        if neighbour_S is not None and neighbour_S.end==False:
            neighbour_S.chance_nodes[0].rewards[self.x + self.y*Y_SIZE] = reward
            neighbour_S.chance_nodes[2].rewards[self.x + self.y*Y_SIZE] = reward
            neighbour_S.chance_nodes[3].rewards[self.x + self.y*Y_SIZE] = reward
        if neighbour_W is not None and neighbour_W.end==False:
            neighbour_W.chance_nodes[0].rewards[self.x + self.y*Y_SIZE] = reward
            neighbour_W.chance_nodes[1].rewards[self.x + self.y*Y_SIZE] = reward
            neighbour_W.chance_nodes[3].rewards[self.x + self.y*Y_SIZE] = reward
        if neighbour_E is not None and neighbour_E.end==False:
            neighbour_E.chance_nodes[0].rewards[self.x + self.y*Y_SIZE] = reward
            neighbour_E.chance_nodes[1].rewards[self.x + self.y*Y_SIZE] = reward
            neighbour_E.chance_nodes[2].rewards[self.x + self.y*Y_SIZE] = reward


        return

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


def compute_utility(start_x,start_y,policy, paint):

    utility =0.0

    current_state= find_state(start_x, start_y, states)
    if paint==True:
        current_state.paint()

    number_of_steps=0

    while(current_state.end==False):
        next_action = policy[current_state.x + current_state.y * Y_SIZE]
        #print(next_action)
        current_state,uti = current_state.next_state(next_action,states)
        if paint == True:
            current_state.paint()

        utility = utility + uti*pow(GAMMA, number_of_steps)
        number_of_steps = number_of_steps+1

    return utility

def find_state(x,y,states):

    result = None

    for s in states:
        if s.x==x and s.y==y:
            result = s

    return result

def sumatori(states,x,y,action,Vt_before):

    state =find_state(x,y,states)

    possible_states = state.get_chance_node(action).possible_next_sates()
    sumatori=0.0

    for possible_state in possible_states:
        sumatori= sumatori + possible_state[1]*(possible_state[2] + GAMMA*Vt_before[possible_state[0].x + possible_state[0].y * Y_SIZE])


    return sumatori

def get_max_and_best_action(sumatoris):

    best_so_far = sumatoris[0]

    if(sumatoris[1][0]>best_so_far[0]):
        best_so_far=sumatoris[1]
    if(sumatoris[2][0]>best_so_far[0]):
        best_so_far=sumatoris[2]
    if(sumatoris[3][0]>best_so_far[0]):
        best_so_far=sumatoris[3]
    return best_so_far

def get_random_policy(states):

    policy= []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if(find_state(x,y,states).end==False):

                policy.append("S")
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


if __name__ == '__main__':

    x_end, y_end = 0,0
    x_end2, y_end2 = 4, 4

    states = []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):
            if (x==x_end and y == y_end) or (x==x_end2 and y == y_end2):
                states.append(State(x,y,True))
            else:
                states.append(State(x, y, False))


    find_state(x_end,y_end,states).set_reward(1,states)
    find_state(x_end2, y_end2, states).set_reward(-1, states)


    # Example to see transition probabilities:
    #from_x, from_y = 1, 0
    #to_x, to_y = 1, 1
    #direction = 0  # 0(N), 1(S), 2(W), 3(E)
    #print(find_state(from_x, from_y, states).chance_nodes[direction].trans_probabilities[to_x + to_y * Y_SIZE])
    #print(find_state(from_x, from_y, states).chance_nodes[direction].rewards[to_x + to_y * Y_SIZE])

    policy_random_example = get_random_policy(states) # is simply an list of strings...

    start_x,start_y = 0,4

    print("What do you want to do?:")
    print("\t0) Compute single utility from a random walk.")
    print("\t1) Compute expected utility of a given policy for a given start state (with a montecarlo method).")
    print("\t2) Compute expected utility of a given policy for all states (with an iteration algorithm).")
    print("\t3) Compute best policy.")

    option=input()

    if option=="0":

        print(compute_utility(start_x, start_y, policy=policy_random_example, paint=True))

    elif option=="1":

        utilities = []
        number_episodes = 100000
        ten_per_cent = number_episodes/10

        for i in range(number_episodes):

            if i%ten_per_cent==0: print("|")

            utilities.append(compute_utility(start_x,start_y,policy=policy_random_example,paint=False))

        estimated_expected_utility = sum(utilities)/ number_episodes
        print(estimated_expected_utility)   # FROM policy_random_example STARTING FROM start_x,start_y!
                                            # Maybe a good idea to compute it from other start states?

    elif option=="2":

        V=[0.0]*NUM_STATES
        Vt_before = [0.0]*NUM_STATES

        for t in range(10000):
            for i in range(X_SIZE):
                for j in range(Y_SIZE):
                    if find_state(i,j,states).end==False:
                        V[i + j*Y_SIZE]= sumatori(states,i,j,policy_random_example[i + j * Y_SIZE],Vt_before)
                        Vt_before[i + j * Y_SIZE]= V[i + j*Y_SIZE]

        for i in range(X_SIZE):
            for j in range(Y_SIZE):
                print(str(i) + " " + str(j) +": "+ str(V[i+j*Y_SIZE]))

    elif option=="3":

        V_opt = [0.0]*NUM_STATES
        Vt_opt_before = [0.0]*NUM_STATES
        policy_optimum = [None]*NUM_STATES

        for t in range(10000):
            for i in range(X_SIZE):
                for j in range(Y_SIZE):
                    if find_state(i, j, states).end == False:

                        sumatoris = []
                        sumatoris.append([sumatori(states, i, j, "N", Vt_opt_before),"N"])
                        sumatoris.append([sumatori(states, i, j, "S", Vt_opt_before),"S"])
                        sumatoris.append([sumatori(states, i, j, "W", Vt_opt_before),"W"])
                        sumatoris.append([sumatori(states, i, j, "E", Vt_opt_before),"E"])

                        V_opt[i + j * Y_SIZE],policy_optimum[i + j * Y_SIZE] = get_max_and_best_action(sumatoris)
                        Vt_opt_before[i + j * Y_SIZE] = V_opt[i + j * Y_SIZE]

        #for i in range(X_SIZE):
        #    for j in range(Y_SIZE):
        #        print(str(i) + " " + str(j) + ": " + str(V_opt[i + j * Y_SIZE]))

        print()
        for y in range(Y_SIZE-1, -1, -1):
            for x in range(X_SIZE):

                if policy_optimum[x+ y*Y_SIZE]!=None:
                    print(policy_optimum[x+ y*Y_SIZE], end=" ")
                else:
                    print(" ",end = " ")
            print()
        print()






    # I després apendre probabilitats (RL) Nou repo???


